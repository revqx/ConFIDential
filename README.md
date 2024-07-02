# fid-flaws

## Preliminary FID Results

| Model                   | Uniform (50 per class) | Uniform (50k times random choice of 1000 classes) | Real (Underlying ImageNet distribution ~50k) |
|-------------------------|------------------------|---------------------------------------------------|----------------------------------------------|
| VAR (seed=42)           | 5.36 | 5.41 | 5.42 |
| MDT (seed=42)           | 2.28 | 2.30 | 2.27 |
| DiT                     |      |      |      |
| LDM                     |      |      |      |
| StyleGAN-XL (seed=1000) | 2.60 | 2.56 | 2.60 |
| StyleGAN-XL (seed=42)   |      |      |      |
| ADM                     |      |      |      |
| MaskedDiT               |      |      |      |

## Preliminary FDD Results (seed=42)