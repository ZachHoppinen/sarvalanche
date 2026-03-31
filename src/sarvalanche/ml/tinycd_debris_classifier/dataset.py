"""TinyCD debris detection dataset.

Same as Siamese — returns pre/post/static/label patches.
Reuses SiameseDebrisDataset directly.
"""

from sarvalanche.ml.siamese_debris_classifier.dataset import (  # noqa: F401
    SiameseDebrisDataset as TinyCDDebrisDataset,
    build_lazy_dataset,
)
