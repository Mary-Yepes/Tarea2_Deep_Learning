AP medio por tipo celular:

| modelo | astro | cort | shsy5y |
|---|---|---|---|
| Baseline clásico | 0.175 | 0.232 | 0.062 |
| Cellpose 2.0 (ajustado) | 0.161 | 0.234 | 0.064 |
| Hover-Net | 0.166 | 0.203 | 0.054 |
| U-Net++ + Watershed | 0.096 | 0.155 | 0.075 |

AP medio por tercil de densidad (número de células):

| modelo | baja | media | alta |
|---|---|---|---|
| Baseline clásico | 0.242 | 0.165 | 0.062 |
| Cellpose 2.0 (ajustado) | 0.229 | 0.166 | 0.064 |
| Hover-Net | 0.232 | 0.137 | 0.054 |
| U-Net++ + Watershed | 0.128 | 0.122 | 0.075 |

Orden por estrato:

| estrato | orden | tau |
|---|---|---|
| global | Baseline clásico > Cellpose 2.0 (ajustado) > Hover-Net > U-Net++ + Watershed | 1 |
| astro | Baseline clásico > Hover-Net > Cellpose 2.0 (ajustado) > U-Net++ + Watershed | 0.6667 |
| cort | Cellpose 2.0 (ajustado) > Baseline clásico > Hover-Net > U-Net++ + Watershed | 0.6667 |
| shsy5y | U-Net++ + Watershed > Cellpose 2.0 (ajustado) > Baseline clásico > Hover-Net | -0.3333 |

**H3.** **Apoyada**: el orden de los modelos cambia al estratificar por tipo celular (τ de Kendall con el global: astro 0.67, cort 0.67, shsy5y -0.33).

**Mejor modelo por tipo:** astro: Baseline clásico; cort: Cellpose 2.0 (ajustado); shsy5y: U-Net++ + Watershed.
