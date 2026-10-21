**augment**

| nivel | mean | std |
|---|---|---|
| geom+photo+elastic | 0.1923 | 0.1395 |
| none | 0.1764 | 0.1278 |

geom+photo+elastic − none: Δ = 0.016 [0.006, 0.027], p = 0.141

**encoder**

| nivel | mean | std |
|---|---|---|
| resnet50_imagenet | 0.2078 | 0.1411 |
| scratch | 0.1923 | 0.1395 |

resnet50_imagenet − scratch: Δ = 0.015 [0.008, 0.022], p = 1.46e-06

**loss**

| nivel | mean | std |
|---|---|---|
| bce | 0.1662 | 0.1262 |
| bce_dice | 0.1923 | 0.1395 |

bce − bce_dice: Δ = -0.026 [-0.035, -0.018], p = 5.01e-09
