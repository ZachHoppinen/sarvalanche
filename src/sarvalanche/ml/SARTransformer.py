import torch
import torch.nn as nn


class SARTransformer(nn.Module):
    def __init__(self, img_size=16, patch_size=8, in_chans=2,
                 embed_dim=256, depth=4, num_heads=4,
                 min_sigma=0.05, max_seq_len=10, dropout=0.2):
        super().__init__()

        self.in_chans    = in_chans
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.min_sigma   = min_sigma
        self.max_seq_len = max_seq_len

        # Number of spatial patches per timestep: (16/8)^2 = 4
        n_patches = (img_size // patch_size) ** 2
        patch_dim = in_chans * patch_size * patch_size  # 2 * 8 * 8 = 128

        # Learned patch embedding
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # Learned spatial positional embedding (4 patches)
        self.spatial_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim))

        # Learned temporal positional embedding (max 10 timesteps)
        self.temporal_embed = nn.Parameter(torch.randn(1, max_seq_len, 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder — operates on all 40 tokens jointly
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.time_pool = nn.Linear(embed_dim, 1)

        # Projection heads: embed_dim → 768 → n_pixels per channel
        out_dim = in_chans * img_size * img_size

        self.head_mu = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, out_dim)
        )

        self.head_sigma = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, out_dim)
        )

    def patchify(self, x):
        """
        Split (B, C, H, W) into (B, n_patches, patch_dim).
        For 16x16 image with 8x8 patches: 4 patches of dim 2*8*8=128
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # Reshape into patches
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)          # (B, nH, nW, C, p, p)
        x = x.reshape(B, (H // p) * (W // p), -1) # (B, n_patches, patch_dim)
        return x

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        n_patches = (H // self.patch_size) ** 2

        # Patchify each timestep and embed
        patches = []
        for t in range(T):
            p = self.patchify(x[:, t])              # (B, n_patches, patch_dim)
            p = self.patch_embed(p)                 # (B, n_patches, embed_dim)
            p = p + self.spatial_embed              # add spatial positional embedding
            p = p + self.temporal_embed[:, t, :, :] # add temporal positional embedding
            patches.append(p)

        # Concatenate all timestep patches → (B, T*n_patches, embed_dim)
        tokens = torch.cat(patches, dim=1)
        tokens = self.dropout(tokens)

        # Transformer over all 40 tokens jointly
        tokens = self.transformer(tokens)           # (B, T*n_patches, embed_dim)

        # Reshape to (B, T, n_patches, embed_dim) then pool
        tokens = tokens.reshape(B, T, n_patches, -1)

        # Average over patches first
        tokens = tokens.mean(dim=2)                 # (B, T, embed_dim)

        # Learnable temporal pooling
        weights = torch.softmax(self.time_pool(tokens).squeeze(-1), dim=1)  # (B, T)
        features = (tokens * weights.unsqueeze(-1)).sum(dim=1)              # (B, embed_dim)

        # Predict
        mu        = self.head_mu(features).reshape(B, self.in_chans, self.img_size, self.img_size)
        sigma_raw = torch.exp(self.head_sigma(features)).reshape(B, self.in_chans, self.img_size, self.img_size)
        sigma     = torch.clamp(sigma_raw, min=self.min_sigma)

        return mu, sigma