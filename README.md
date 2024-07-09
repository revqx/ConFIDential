# fid-flaws

## Preliminary FID Results

| Model                   | Uniform (50 per class) | Uniform (50k times random choice of 1000 classes) | Real (Underlying ImageNet distribution ~50k) |
|-------------------------|------------------------|---------------------------------------------------|----------------------------------------------|
| VAR (seed=42)           | 5.36 | 5.41 | 5.42 |
| MDT (seed=42)           | 2.28 | 2.30 | 2.27 |
| DiT                     | 2.82 | 2.83 | 2.79 |
| LDM                     | 3.56 | 3.54 | 3.53 |
| StyleGAN-XL (seed=1000) | 2.60 | 2.56 | 2.60 |
| StyleGAN-XL (seed=42)   | 2.61 | 2.55 | 2.56 |
| ADM                     |      |      |      |
| MaskedDiT               | 2.32 | 2.34 | 2.30 |

## Preliminary FDD Results (seed=42)