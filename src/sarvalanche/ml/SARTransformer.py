import torch
import torch.nn as nn
import timm  # PyTorch Image Models library


class SARTransformer(nn.Module):
    def __init__(self, img_size=16, patch_size=8, in_chans=1, max_seq_len=10,
                 embed_dim=256, depth=4, num_heads=4, min_sigma=0.05):
        super().__init__()

        self.in_chans = in_chans
        self.img_size = img_size
        self.min_sigma = min_sigma
        self.max_seq_len = max_seq_len

        # Use timm's Vision Transformer backbone
        self.vit = timm.create_model(
            'vit_small_patch8_224',
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # no classification head
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        # Add temporal positional embeddings
        self.temporal_embed = nn.Parameter(
            torch.randn(1, self.max_seq_len, embed_dim)
        )
        self.time_pool = nn.Linear(embed_dim, 1)

        # Projection heads - use in_chans for output size
        self.head_mu = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.Linear(768, in_chans * img_size * img_size)
        )

        self.head_sigma = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.Linear(768, in_chans * img_size * img_size)
        )

    def forward(self, x):
        # x: (batch, time, channels, height, width)
        B, T, C, H, W = x.shape

        # Process each timestep through ViT
        features = []
        for t in range(T):
            feat = self.vit.forward_features(x[:, t])  # (B, num_patches, embed_dim)
            # feat = feat.mean(dim=1)  # global average pooling → (B, embed_dim)
            feat = feat[:, 0]  # use CLS token → (B, embed_dim)
            features.append(feat)

        features = torch.stack(features, dim=1)  # (B, T, embed_dim)

        # Add temporal embeddings
        features = features + self.temporal_embed[:, :T, :]

        # Learnable weighted pool over time
        weights  = torch.softmax(self.time_pool(features).squeeze(-1), dim=1)  # (B, T)
        features = (features * weights.unsqueeze(-1)).sum(dim=1)               # (B, embed_dim)

        # Predict mean and std with correct shape
        mu = self.head_mu(features).reshape(B, self.in_chans, self.img_size, self.img_size)
        sigma_raw = torch.exp(self.head_sigma(features)).reshape(B, self.in_chans, self.img_size, self.img_size)

        # Clamp sigma in the forward pass itself
        sigma = torch.clamp(sigma_raw, min=self.min_sigma)


        return mu, sigma
