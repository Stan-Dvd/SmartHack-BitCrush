| Model | MAE (rând) | RMSE (rând) | R² (rând) | MAE ore încărcare/zi (stație) | Spearman cerere zilnică |
|---|---|---|---|---|---|
| B0 (profil oraș) | 3.373 | 5.796 | 0.005 | 76.84 | — |
| B1 (ocupare × n_piles) | 2.035 | 3.849 | 0.561 | 44.05 | 0.763 |
| LightGBM | 1.865 | 3.283 | 0.681 | 39.85 | 0.758 |

Set de test: 14,816 rânduri, 309 stații, 63 zone (metricile pe stație: 307 stații cu toate cele 48 de celule). Media lui y în test: 3.874; ore de încărcare pe zi, în medie: 90.6.
