# Tensión de liquidez · Random Forest

Aplicación Streamlit que entrena un Random Forest para predecir si una empresa o proyecto caerá en tensión de liquidez (`Tension_Liquidez_bin`), a partir de variables de capital de trabajo, liquidez y deuda de corto plazo.

## Contenido del repositorio

| Archivo | Función |
|---|---|
| `app.py` | Aplicación completa: ingeniería de variables, entrenamiento, evaluación, simulador, carga por lotes y exportación del modelo |
| `requirements.txt` | Dependencias con versiones fijadas |
| `README.md` | Este documento |

**La base de datos no forma parte del repositorio.** La aplicación la solicita al iniciar y la procesa solo en memoria durante la sesión; no se guarda en el servidor ni en GitHub.

## Ejecución local

Requiere Python 3.11 o superior.

```bash
git clone <url-del-repositorio>
cd <carpeta-del-repositorio>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

La app abre en `http://localhost:8501`.

## Publicación en Streamlit Community Cloud

1. En [share.streamlit.io](https://share.streamlit.io), crea una app nueva apuntando a este repositorio, rama principal y archivo `app.py`.
2. En **Advanced settings**, selecciona Python 3.12.
3. Despliega. Si el repositorio es privado, la app puede restringirse a usuarios invitados desde la configuración de compartir.

## Uso

1. **Sube la base de datos** en la barra lateral (Excel o CSV). Si es Excel, se lee la hoja `Datos_Modelo`; si no existe, la primera hoja.
2. El modelo se entrena automáticamente: 30 combinaciones de hiperparámetros × 5 folds de validación cruzada. Tarda de 20 segundos a 2 minutos según el equipo o servidor; después queda en caché mientras no cambie el archivo.
3. Ajusta el **umbral de decisión** en la barra lateral. Las métricas se recalculan sin reentrenar.

Pestañas disponibles:

- **Desempeño.** ROC-AUC en prueba, validación cruzada y entrenamiento; recall, precisión, F1, balanced accuracy y accuracy en el umbral elegido; matriz de confusión con lectura para tesorería; hiperparámetros seleccionados.
- **Análisis por sector.** Tensión real, probabilidad promedio predicha, alertas emitidas y métricas por sector; señala sectores con menos de 30 casos de prueba.
- **Simulador.** Escenario individual con montos en millones y porcentajes en %. CxC, inventarios, CxP, capital de trabajo neto, ciclo de conversión y EBITDA se calculan a partir de ventas, días y márgenes para mantener la consistencia contable.
- **Carga por lotes.** Descarga la plantilla, llénala y súbela; obtienes probabilidad y clasificación por caso en Excel. Si el archivo trae `Tension_Liquidez_bin`, se reporta el desempeño.
- **Datos.** Tamaño de la base, balance de clases y variables usadas y excluidas.

## Estructura requerida de la base

| Tipo | Columnas (nombres exactos) |
|---|---|
| Objetivo | `Tension_Liquidez_bin` (0 = sin tensión, 1 = con tensión) |
| Fecha / categórica | `Fecha`, `Sector` |
| Numéricas (20) | `Ventas_12M_MXN`, `Crecimiento_Ventas_pct`, `Margen_EBITDA_pct`, `DSO_dias`, `DIO_dias`, `DPO_dias`, `CCC_dias`, `CxC_MXN`, `Inventarios_MXN`, `CxP_MXN`, `NWC_MXN`, `Deuda_CP_MXN`, `Caja_MXN`, `Linea_Credito_Disponible_MXN`, `Concentracion_Top5_Clientes_pct`, `Morosidad_CxC_pct`, `Inventario_Obsoleto_pct`, `Volatilidad_Ventas_pct`, `Estacionalidad_bin`, `EBITDA_MXN` |

Los porcentajes van en decimal (15% = 0.15). Las columnas adicionales se ignoran. `ID_Observacion`, `Prob_Tension_Liquidez` y `Brecha_Caja_90d_MXN` se excluyen del entrenamiento aunque vengan en el archivo (las dos últimas por *leakage*).

## Configuración del modelo

| Componente | Configuración |
|---|---|
| Algoritmo | `RandomForestClassifier` dentro de un `Pipeline` de scikit-learn |
| Predictores | 20 numéricos + `Sector` (one-hot) + `Mes` y `Trimestre` derivados de `Fecha` + 4 razones: Caja / Deuda CP, (Caja + Línea) / Deuda CP, NWC / Ventas, Deuda CP / EBITDA |
| Partición | Aleatoria estratificada 70/30, `random_state=42` |
| Hiperparámetros | `RandomizedSearchCV`, 30 combinaciones, `StratifiedKFold` de 5 folds, optimizando ROC-AUC |
| Espacio de búsqueda | `n_estimators` 200–800 · `max_depth` {None, 4, 6, 8, 10, 14} · `min_samples_leaf` {1, 2, 4, 8, 12} · `max_features` {sqrt, log2, 0.3, 0.5} · `class_weight` {None, balanced, balanced_subsample} |
| Umbral | Ajustable 0.05–0.95 (inicio 0.50) |

Las razones con denominador cero o negativo toman un tope de 99 (sin deuda CP = cobertura máxima; EBITDA ≤ 0 = apalancamiento máximo).

## Reutilizar el modelo exportado

El botón **Descargar modelo (.joblib)** entrega el pipeline completo y el umbral vigente. Para usarlo desde otro script, coloca `app.py` en la misma carpeta (o en el `PYTHONPATH`) e impórtalo antes de cargar el modelo; al importarlo no se ejecuta la interfaz.

```python
import app      # registra la ingeniería de variables que usa el modelo
import joblib
import pandas as pd

paquete = joblib.load("rf_tension_liquidez_AAAAMMDD.joblib")
modelo, umbral = paquete["modelo"], paquete["umbral"]

casos = pd.read_excel("casos_nuevos.xlsx")
casos["prob_tension"] = modelo.predict_proba(casos[paquete["columnas_requeridas"]])[:, 1]
casos["alerta"] = (casos["prob_tension"] >= umbral).astype(int)
```

El paquete incluye además `mejores_parametros`, `metricas`, `fecha_entrenamiento` y `version_sklearn`. Cárgalo con la misma versión de scikit-learn con la que se entrenó.

## Limitaciones

- La partición aleatoria mezcla periodos; en uso real se predice hacia adelante, por lo que el desempeño productivo puede ser menor al reportado.
- Las probabilidades del Random Forest ordenan bien el riesgo pero no están calibradas: 70% no equivale necesariamente a 70% de frecuencia observada.
- Sectores con pocos casos de prueba tienen métricas inestables.
- La app no incluye importancia de variables ni explicación por predicción.
