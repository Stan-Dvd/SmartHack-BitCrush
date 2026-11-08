| Model | MAE (rând) | RMSE (rând) | R² (rând) | MAE ore încărcare/zi (stație) | Spearman cerere zilnică |
|---|---|---|---|---|---|
| B0 (profil oraș) | 3.269 | 5.334 | 0.008 | 74.26 | — |
| B1 (ocupare × n_piles) | 1.930 | 3.400 | 0.597 | 41.59 | 0.775 |
| LightGBM | 1.795 | 2.949 | 0.697 | 39.05 | 0.768 |

Set de test: 14,576 rânduri, 304 stații, 61 zone (metricile pe stație: 302 stații cu toate cele 48 de celule). Media lui y în test: 3.757; ore de încărcare pe zi, în medie: 87.8.
