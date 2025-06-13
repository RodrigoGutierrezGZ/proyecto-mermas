# Conclusión Final: Comparativa de Modelos con Ingeniería de Características

Se evaluaron tres modelos utilizando características avanzadas derivadas de las columnas de merma.

## Métricas Comparativas

| Modelo           |     R² |      RMSE |      MAE |   MAPE (%) |   Tiempo (s) |
|:-----------------|-------:|----------:|---------:|-----------:|-------------:|
| Regresión Lineal | 1      |    0.0206 |   0.0031 |     0.0004 |       0.2425 |
| Random Forest    | 0.8855 | 1444.21   |  48.6549 |     0.1008 |       1.6853 |
| LightGBM         | 0.6117 | 2659.53   | 221.08   |     3.6002 |       0.2471 |

El mejor modelo basado en R² fue **Regresión Lineal**.

## Muestra de Predicciones del Mejor Modelo (Regresión Lineal)

|   Valor real |   Predicción Regresión Lineal |   Error % Regresión Lineal | categoria                 | tienda     | tipo_movimiento   |
|-------------:|------------------------------:|---------------------------:|:--------------------------|:-----------|:------------------|
|         -960 |                     -959.9969 |                     0.0003 | CECINAS GRANEL            | TEMUCO IV  | Pérdida           |
|        -1368 |                    -1368.0006 |                     0.0000 | PAPEL HIGIENICOS          | TEMUCO III | Pérdida           |
|         -371 |                     -371.0010 |                     0.0003 | CECINAS GRANEL            | TEMUCO III | Pérdida           |
|        -3005 |                    -3005.0012 |                     0.0000 | AZUCAR                    | TEMUCO III | Pérdida           |
|         -169 |                     -169.0002 |                     0.0001 | POMACEAS                  | ANGOL      | Pérdida           |
|         -546 |                     -545.9999 |                     0.0000 | GALLETAS                  | TEMUCO III | Pérdida           |
|         -246 |                     -246.0015 |                     0.0006 | CECINAS GRANEL            | ANGOL      | Pérdida           |
|         -313 |                     -313.0013 |                     0.0004 | YOGHURT                   | TEMUCO IV  | Pérdida           |
|        -2651 |                    -2651.0021 |                     0.0001 | PRODUCTOS PARA REPOSTERIA | TEMUCO V   | Pérdida           |
|        -4832 |                    -4831.9758 |                     0.0005 | GALLETAS                  | TEMUCO III | Pérdida           |
|         -253 |                     -252.9985 |                     0.0006 | LECHES SABORES            | TEMUCO IV  | Pérdida           |
|        -1050 |                    -1050.0012 |                     0.0001 | LEGUMBRES                 | ANGOL      | Pérdida           |
|         -886 |                     -886.0004 |                     0.0000 | ARROZ                     | ANGOL      | Pérdida           |
|         -925 |                     -925.0007 |                     0.0001 | CECINAS GRANEL            | TEMUCO IV  | Pérdida           |
|        -5097 |                    -5096.9977 |                     0.0000 | PASTA DENTAL              | TEMUCO V   | Pérdida           |
|         -355 |                     -354.9997 |                     0.0001 | FIDEOS Y PASTAS           | TEMUCO V   | Pérdida           |
|         -279 |                     -279.0015 |                     0.0005 | YOGHURT                   | ANGOL      | Pérdida           |
|         -203 |                     -203.0007 |                     0.0004 | CHOCOLATES                | TEMUCO II  | Pérdida           |
|         -329 |                     -328.9979 |                     0.0007 | CECINAS GRANEL            | TEMUCO II  | Pérdida           |
|         -158 |                     -158.0001 |                     0.0001 | GALLETAS                  | TEMUCO III | Pérdida           |