"""
app.py
------
Random Forest para predecir tensión de liquidez (clasificación binaria)
a partir de variables de capital de trabajo.

Archivo único: contiene la ingeniería de variables, el entrenamiento y la
interfaz Streamlit. La base de datos NO vive en el repositorio; se carga
desde la aplicación.

Ejecutar:  streamlit run app.py
"""

import io
import sys
import types
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# ===========================================================================
# INGENIERÍA DE VARIABLES
# ===========================================================================
# ---------------------------------------------------------------------------
# Definición de columnas
# ---------------------------------------------------------------------------
TARGET = "Tension_Liquidez_bin"

# Columnas que nunca entran al modelo
COLUMNAS_EXCLUIDAS = [
    "ID_Observacion",          # identificador sin contenido predictivo
    "Prob_Tension_Liquidez",   # leakage: es la probabilidad que genera el target
    "Brecha_Caja_90d_MXN",     # leakage: resultado futuro / target de regresión
]

COLUMNA_FECHA = "Fecha"
COLUMNA_CATEGORICA = "Sector"

# 20 predictores numéricos del diccionario
NUMERICAS_BASE = [
    "Ventas_12M_MXN", "Crecimiento_Ventas_pct", "Margen_EBITDA_pct",
    "DSO_dias", "DIO_dias", "DPO_dias", "CCC_dias",
    "CxC_MXN", "Inventarios_MXN", "CxP_MXN", "NWC_MXN",
    "Deuda_CP_MXN", "Caja_MXN", "Linea_Credito_Disponible_MXN",
    "Concentracion_Top5_Clientes_pct", "Morosidad_CxC_pct",
    "Inventario_Obsoleto_pct", "Volatilidad_Ventas_pct",
    "Estacionalidad_bin", "EBITDA_MXN",
]

# Variables derivadas
VARIABLES_FECHA = ["Mes", "Trimestre"]
RAZONES = [
    "Cobertura_Caja_DeudaCP",      # Caja / Deuda CP
    "Liquidez_Disponible_DeudaCP", # (Caja + Línea disponible) / Deuda CP
    "NWC_sobre_Ventas",            # NWC / Ventas 12M
    "DeudaCP_sobre_EBITDA",        # Deuda CP / EBITDA
]

NUMERICAS_MODELO = NUMERICAS_BASE + VARIABLES_FECHA + RAZONES

# Columnas que debe traer un archivo para poder calificarse
COLUMNAS_REQUERIDAS = [COLUMNA_FECHA, COLUMNA_CATEGORICA] + NUMERICAS_BASE

# Tope para razones cuando el denominador es cero o negativo.
# Random Forest es invariante a transformaciones monótonas, así que el valor
# exacto del tope no altera el orden de las divisiones, solo evita infinitos.
TOPE_RAZON = 99.0


def _division_segura(num, den, valor_si_den_no_positivo):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(den > 0, num / den, valor_si_den_no_positivo)
    return np.clip(res, -TOPE_RAZON, TOPE_RAZON)


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame con las columnas originales (al menos COLUMNAS_REQUERIDAS)
    y regresa uno con Sector + las 27 variables numéricas del modelo.
    Se usa como primer paso del Pipeline (FunctionTransformer).
    """
    X = df.copy()

    # --- Variables de fecha -------------------------------------------------
    fecha = pd.to_datetime(X[COLUMNA_FECHA], errors="coerce")
    X["Mes"] = fecha.dt.month
    X["Trimestre"] = fecha.dt.quarter

    # --- Razones financieras -----------------------------------------------
    # Sin deuda CP la cobertura es "excelente" -> tope superior
    X["Cobertura_Caja_DeudaCP"] = _division_segura(
        X["Caja_MXN"], X["Deuda_CP_MXN"], TOPE_RAZON)
    X["Liquidez_Disponible_DeudaCP"] = _division_segura(
        X["Caja_MXN"] + X["Linea_Credito_Disponible_MXN"], X["Deuda_CP_MXN"], TOPE_RAZON)
    X["NWC_sobre_Ventas"] = _division_segura(
        X["NWC_MXN"], X["Ventas_12M_MXN"], 0.0)
    # EBITDA <= 0 es señal de riesgo máximo -> tope superior
    X["DeudaCP_sobre_EBITDA"] = _division_segura(
        X["Deuda_CP_MXN"], X["EBITDA_MXN"], TOPE_RAZON)

    X[COLUMNA_CATEGORICA] = X[COLUMNA_CATEGORICA].astype(str)
    return X[[COLUMNA_CATEGORICA] + NUMERICAS_MODELO]


def validar_columnas(df: pd.DataFrame, incluir_target: bool = False) -> list:
    """Regresa la lista de columnas faltantes (vacía si el archivo es válido)."""
    requeridas = COLUMNAS_REQUERIDAS + ([TARGET] if incluir_target else [])
    return [c for c in requeridas if c not in df.columns]

# Registro del módulo como "app".
# Streamlit ejecuta este archivo como "__main__"; sin este registro el modelo
# exportado en .joblib quedaría ligado a "__main__" y no podría cargarse desde
# otro script. Así, basta con `import app` antes de `joblib.load(...)`.
_modulo_app = sys.modules.get("app") or types.ModuleType("app")
_modulo_app.construir_features = construir_features
sys.modules["app"] = _modulo_app
construir_features.__module__ = "app"


# ===========================================================================
# CONFIGURACIÓN, ENTRENAMIENTO E INTERFAZ
# ===========================================================================
# ---------------------------------------------------------------------------
# Configuración del modelo (acordada)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5
N_ITER_BUSQUEDA = 30          # combinaciones evaluadas en RandomizedSearchCV
METRICA_BUSQUEDA = "roc_auc"
UMBRAL_INICIAL = 0.50
N_MIN_SECTOR = 30             # por debajo de esto, métricas por sector poco estables
BRECHA_SOBREAJUSTE = 0.05     # diferencia AUC entrenamiento-CV que dispara alerta

ESPACIO_BUSQUEDA = {
    "rf__n_estimators": randint(200, 801),
    "rf__max_depth": [None, 4, 6, 8, 10, 14],
    "rf__min_samples_leaf": [1, 2, 4, 8, 12],
    "rf__max_features": ["sqrt", "log2", 0.3, 0.5],
    "rf__class_weight": [None, "balanced", "balanced_subsample"],
}

HOJA_DATOS = "Datos_Modelo"

# Paleta
C_TENSION = "#A63D40"
C_SIN_TENSION = "#3E6B89"
C_NEUTRO = "#9AA5B1"

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def leer_archivo(contenido: bytes, nombre: str) -> pd.DataFrame:
    buffer = io.BytesIO(contenido)
    if nombre.lower().endswith(".csv"):
        return pd.read_csv(buffer)
    hojas = pd.ExcelFile(buffer).sheet_names
    hoja = HOJA_DATOS if HOJA_DATOS in hojas else hojas[0]
    buffer.seek(0)
    return pd.read_excel(buffer, sheet_name=hoja)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def construir_pipeline() -> Pipeline:
    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             [COLUMNA_CATEGORICA]),
            ("num", "passthrough", NUMERICAS_MODELO),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("features", FunctionTransformer(construir_features, validate=False)),
        ("prep", preprocesador),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


@st.cache_resource(show_spinner=False)
def entrenar(df: pd.DataFrame) -> dict:
    datos = df.dropna(subset=[TARGET]).copy()
    X = datos[COLUMNAS_REQUERIDAS]
    y = datos[TARGET].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    busqueda = RandomizedSearchCV(
        estimator=construir_pipeline(),
        param_distributions=ESPACIO_BUSQUEDA,
        n_iter=N_ITER_BUSQUEDA,
        scoring=METRICA_BUSQUEDA,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    busqueda.fit(X_tr, y_tr)
    modelo = busqueda.best_estimator_

    prob_tr = modelo.predict_proba(X_tr)[:, 1]
    prob_te = modelo.predict_proba(X_te)[:, 1]
    idx = busqueda.best_index_

    return {
        "modelo": modelo,
        "X_test": X_te.reset_index(drop=True),
        "y_test": y_te.reset_index(drop=True),
        "prob_test": prob_te,
        "auc_train": roc_auc_score(y_tr, prob_tr),
        "auc_cv": busqueda.best_score_,
        "auc_cv_std": busqueda.cv_results_["std_test_score"][idx],
        "auc_test": roc_auc_score(y_te, prob_te),
        "mejores_parametros": {k.replace("rf__", ""): v
                               for k, v in busqueda.best_params_.items()},
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "tasa_tension": y.mean(),
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "referencia": datos,  # para valores por defecto del simulador
    }


def metricas_en_umbral(y, prob, umbral) -> dict:
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "pred": pred,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def pct(x, dec=1):
    return f"{x * 100:.{dec}f}%"



def main():
    # ---------------------------------------------------------------------------
    # Barra lateral
    # ---------------------------------------------------------------------------
    st.set_page_config(page_title="Tensión de liquidez · Random Forest",
                       page_icon="💧", layout="wide")

    st.sidebar.header("Datos de entrenamiento")
    archivo = st.sidebar.file_uploader(
        "Sube la base de datos (Excel o CSV)", type=["xlsx", "xls", "csv"],
        key="entrenamiento",
        help="Si es Excel, se lee la hoja Datos_Modelo; si no existe, la primera hoja. "
             "El archivo se procesa solo durante la sesión y no se guarda en el servidor.")
    df = leer_archivo(archivo.getvalue(), archivo.name) if archivo is not None else None

    st.sidebar.header("Umbral de decisión")
    umbral = st.sidebar.slider(
        "Probabilidad a partir de la cual se clasifica como tensión",
        min_value=0.05, max_value=0.95, value=UMBRAL_INICIAL, step=0.01,
        help="Bajarlo genera más alertas (detecta más tensiones, más falsas alarmas). "
             "Subirlo reduce falsas alarmas pero deja escapar más casos. "
             "Mover el umbral no reentrena el modelo.")

    with st.sidebar.expander("Configuración de la búsqueda"):
        st.markdown(
            f"- Partición: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} "
            f"estratificada, random_state={RANDOM_STATE}\n"
            f"- Validación cruzada: {CV_FOLDS} folds estratificados\n"
            f"- Combinaciones evaluadas: {N_ITER_BUSQUEDA}\n"
            f"- Métrica optimizada: ROC-AUC\n"
            f"- Predictores: {len(NUMERICAS_MODELO)} numéricos + Sector")

    # ---------------------------------------------------------------------------
    # Encabezado
    # ---------------------------------------------------------------------------
    st.title("Tensión de liquidez")
    st.caption("Clasificación con Random Forest a partir de variables de capital de trabajo")

    if df is None:
        st.info("Sube la base de datos en la barra lateral para entrenar el modelo.")
        st.markdown(
            "**Estructura requerida**\n\n"
            f"- Variable objetivo: `{TARGET}` (0 = sin tensión, 1 = con tensión)\n"
            f"- Fecha y categórica: `{COLUMNA_FECHA}`, `{COLUMNA_CATEGORICA}`\n"
            f"- Numéricas ({len(NUMERICAS_BASE)}): " +
            ", ".join(f"`{c}`" for c in NUMERICAS_BASE) + "\n"
            "- Porcentajes en decimal (15% = 0.15). Columnas adicionales se ignoran; "
            + ", ".join(f"`{c}`" for c in COLUMNAS_EXCLUIDAS) +
            " se excluyen del entrenamiento aunque vengan en el archivo.")
        st.stop()

    faltantes = validar_columnas(df, incluir_target=True)
    if faltantes:
        st.error("Al archivo le faltan estas columnas: " + ", ".join(faltantes))
        st.stop()

    with st.spinner(f"Entrenando: {N_ITER_BUSQUEDA} combinaciones × {CV_FOLDS} folds…"):
        res = entrenar(df)

    m = metricas_en_umbral(res["y_test"], res["prob_test"], umbral)

    # ---------------------------------------------------------------------------
    # Exportar modelo (barra lateral, requiere modelo entrenado)
    # ---------------------------------------------------------------------------
    st.sidebar.header("Exportar modelo")
    paquete = {
        "modelo": res["modelo"],
        "umbral": umbral,
        "columnas_requeridas": COLUMNAS_REQUERIDAS,
        "mejores_parametros": res["mejores_parametros"],
        "metricas": {"auc_cv": res["auc_cv"], "auc_test": res["auc_test"],
                     "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]},
        "fecha_entrenamiento": res["fecha_entrenamiento"],
        "version_sklearn": sklearn.__version__,
    }
    buf_modelo = io.BytesIO()
    joblib.dump(paquete, buf_modelo)
    st.sidebar.download_button(
        "Descargar modelo (.joblib)", buf_modelo.getvalue(),
        file_name=f"rf_tension_liquidez_{date.today():%Y%m%d}.joblib",
        mime="application/octet-stream",
        help="Incluye el pipeline completo y el umbral actual. "
             "Para cargarlo se necesita features.py en la misma carpeta.")

    tab_desemp, tab_sector, tab_sim, tab_lotes, tab_datos = st.tabs(
        ["Desempeño", "Análisis por sector", "Simulador", "Carga por lotes", "Datos"])

    # ---------------------------------------------------------------------------
    # 1. Desempeño
    # ---------------------------------------------------------------------------
    with tab_desemp:
        st.subheader("Capacidad de discriminación")
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC prueba", f"{res['auc_test']:.3f}",
                  help="Probabilidad de que el modelo asigne más riesgo a un caso con "
                       "tensión que a uno sin tensión. 0.5 = azar, 1.0 = perfecto.")
        c2.metric("ROC-AUC validación cruzada", f"{res['auc_cv']:.3f}",
                  f"± {res['auc_cv_std']:.3f}", delta_color="off")
        c3.metric("ROC-AUC entrenamiento", f"{res['auc_train']:.3f}")

        st.caption("En Random Forest es normal que el AUC de entrenamiento esté cerca de 1.0, "
                   "porque cada árbol ve casi todos sus datos. La prueba de estabilidad "
                   "relevante es la comparación entre validación cruzada y prueba.")
        if abs(res["auc_cv"] - res["auc_test"]) <= BRECHA_SOBREAJUSTE:
            st.success("El AUC de prueba es consistente con la validación cruzada: "
                       "el desempeño es estable fuera de muestra.")
        else:
            st.info("El AUC de prueba difiere más de 0.05 del de validación cruzada. "
                    "Con pocos casos de prueba es una variación posible; conviene revisar "
                    "con más datos.")

        st.subheader(f"Clasificación con umbral {umbral:.2f}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Recall", pct(m["recall"]),
                  help="De las tensiones reales, qué porcentaje detecta el modelo.")
        c2.metric("Precisión", pct(m["precision"]),
                  help="De las alertas emitidas, qué porcentaje fueron tensión real.")
        c3.metric("F1", f"{m['f1']:.3f}")
        c4.metric("Balanced accuracy", pct(m["balanced_accuracy"]))
        c5.metric("Accuracy", pct(m["accuracy"]))

        col_cm, col_txt = st.columns([3, 2])
        with col_cm:
            z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
            etiquetas = [[f"Sin tensión bien identificada<br><b>{m['tn']}</b>",
                          f"Falsa alarma<br><b>{m['fp']}</b>"],
                         [f"Tensión no detectada<br><b>{m['fn']}</b>",
                          f"Tensión detectada<br><b>{m['tp']}</b>"]]
            fig_cm = go.Figure(go.Heatmap(
                z=z, x=["Predice sin tensión", "Predice tensión"],
                y=["Real: sin tensión", "Real: tensión"],
                text=etiquetas, texttemplate="%{text}", textfont={"size": 14},
                colorscale=[[0, "#F2F4F7"], [1, C_SIN_TENSION]], showscale=False,
                hovertemplate="%{y} · %{x}: %{z}<extra></extra>"))
            fig_cm.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                                 yaxis=dict(autorange="reversed"),
                                 title=f"Matriz de confusión · {res['n_test']} casos de prueba")
            st.plotly_chart(fig_cm, width="stretch")
        with col_txt:
            st.markdown("**Lectura para tesorería**")
            st.markdown(
                f"- De **{m['tp'] + m['fn']}** casos con tensión real, el modelo anticipa "
                f"**{m['tp']}** y deja escapar **{m['fn']}**.\n"
                f"- Emite **{m['tp'] + m['fp']}** alertas; **{m['fp']}** no se materializan.\n"
                f"- Cada tensión no detectada es un caso sin margen para disponer de línea "
                f"o renegociar pagos; cada falsa alarma cuesta una revisión preventiva.")
            st.markdown("**Hiperparámetros seleccionados**")
            st.dataframe(pd.DataFrame(
                [(k, str(v)) for k, v in res["mejores_parametros"].items()],
                columns=["Parámetro", "Valor"]), hide_index=True, width="stretch")

    # ---------------------------------------------------------------------------
    # 2. Análisis por sector
    # ---------------------------------------------------------------------------
    with tab_sector:
        st.subheader(f"Resultados por sector en el conjunto de prueba · umbral {umbral:.2f}")
        base = res["X_test"][[COLUMNA_CATEGORICA]].copy()
        base["real"] = res["y_test"].values
        base["prob"] = res["prob_test"]
        base["pred"] = m["pred"]

        filas = []
        for sector, g in base.groupby(COLUMNA_CATEGORICA):
            auc = roc_auc_score(g["real"], g["prob"]) if g["real"].nunique() == 2 else np.nan
            filas.append({
                "Sector": sector,
                "Casos": len(g),
                "Tensión real": g["real"].mean(),
                "Prob. promedio predicha": g["prob"].mean(),
                "Alertas emitidas": g["pred"].mean(),
                "Recall": recall_score(g["real"], g["pred"], zero_division=0),
                "Precisión": precision_score(g["real"], g["pred"], zero_division=0),
                "F1": f1_score(g["real"], g["pred"], zero_division=0),
                "ROC-AUC": auc,
            })
        tabla = pd.DataFrame(filas).sort_values("Tensión real", ascending=False)

        fig_s = go.Figure()
        fig_s.add_bar(x=tabla["Sector"], y=tabla["Tensión real"], name="Tensión real",
                      marker_color=C_TENSION,
                      text=[pct(v, 0) for v in tabla["Tensión real"]], textposition="outside")
        fig_s.add_bar(x=tabla["Sector"], y=tabla["Prob. promedio predicha"],
                      name="Probabilidad promedio predicha", marker_color=C_SIN_TENSION,
                      text=[pct(v, 0) for v in tabla["Prob. promedio predicha"]],
                      textposition="outside")
        fig_s.add_bar(x=tabla["Sector"], y=tabla["Alertas emitidas"],
                      name=f"Alertas emitidas (umbral {umbral:.2f})", marker_color=C_NEUTRO,
                      text=[pct(v, 0) for v in tabla["Alertas emitidas"]],
                      textposition="outside")
        fig_s.update_layout(barmode="group", height=420, yaxis_tickformat=".0%",
                            yaxis_range=[0, 1.1], legend=dict(orientation="h", y=1.12),
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_s, width="stretch")
        st.caption("Si la probabilidad promedio predicha se aleja mucho de la tensión real "
                   "de un sector, el modelo está sobre o subestimando el riesgo de ese sector.")

        formato = {"Tensión real": "{:.1%}", "Prob. promedio predicha": "{:.1%}",
                   "Alertas emitidas": "{:.1%}", "Recall": "{:.1%}", "Precisión": "{:.1%}",
                   "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}
        st.dataframe(tabla.style.format(formato, na_rep="n/d"),
                     hide_index=True, width="stretch")

        pequenos = tabla.loc[tabla["Casos"] < N_MIN_SECTOR, "Sector"].tolist()
        if pequenos:
            st.warning(f"{', '.join(pequenos)}: menos de {N_MIN_SECTOR} casos de prueba. "
                       "Sus métricas pueden cambiar mucho con pocos casos; úsalas como "
                       "referencia, no como conclusión.")

    # ---------------------------------------------------------------------------
    # 3. Simulador
    # ---------------------------------------------------------------------------
    with tab_sim:
        ref = res["referencia"]
        med = ref[NUMERICAS_BASE].median()
        # Razón costo de ventas / ventas implícita en la base (vía DIO e inventarios)
        mask = ref["DIO_dias"] > 0
        ratio_costo = float((ref.loc[mask, "Inventarios_MXN"] / ref.loc[mask, "DIO_dias"] * 365
                             / ref.loc[mask, "Ventas_12M_MXN"]).median()) if mask.any() else 0.62
        MM = 1_000_000

        st.subheader("Escenario individual")
        st.caption("Los valores iniciales son la mediana de la base de entrenamiento. "
                   "Montos en millones de pesos; porcentajes en %.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Perfil**")
            sectores = sorted(ref[COLUMNA_CATEGORICA].astype(str).unique())
            s_sector = st.selectbox("Sector", sectores)
            s_fecha = st.date_input("Fecha de evaluación", value=date.today())
            s_ventas = st.number_input("Ventas 12M (millones)", min_value=0.1,
                                       value=round(med["Ventas_12M_MXN"] / MM, 2), step=0.5)
            s_crec = st.number_input("Crecimiento de ventas (%)",
                                     value=round(med["Crecimiento_Ventas_pct"] * 100, 1), step=1.0)
            s_margen = st.number_input("Margen EBITDA (%)",
                                       value=round(med["Margen_EBITDA_pct"] * 100, 1), step=0.5)
            s_vol = st.number_input("Volatilidad de ventas (%)", min_value=0.0,
                                    value=round(med["Volatilidad_Ventas_pct"] * 100, 1), step=1.0)
            s_est = st.toggle("Negocio estacional", value=bool(med["Estacionalidad_bin"] >= 0.5))
        with c2:
            st.markdown("**Ciclo de capital de trabajo**")
            s_dso = st.number_input("DSO (días)", min_value=0.0,
                                    value=round(med["DSO_dias"], 0), step=1.0)
            s_dio = st.number_input("DIO (días)", min_value=0.0,
                                    value=round(med["DIO_dias"], 0), step=1.0)
            s_dpo = st.number_input("DPO (días)", min_value=0.0,
                                    value=round(med["DPO_dias"], 0), step=1.0)
            s_costo = st.number_input("Costo de ventas / ventas (%)", min_value=0.0,
                                      max_value=100.0, value=round(ratio_costo * 100, 1), step=1.0,
                                      help="Base para convertir DIO y DPO en inventarios y "
                                           "cuentas por pagar.")
        with c3:
            st.markdown("**Liquidez y deuda**")
            s_caja = st.number_input("Caja (millones)", min_value=0.0,
                                     value=round(med["Caja_MXN"] / MM, 2), step=0.1)
            s_linea = st.number_input("Línea de crédito disponible (millones)", min_value=0.0,
                                      value=round(med["Linea_Credito_Disponible_MXN"] / MM, 2),
                                      step=0.1)
            s_deuda = st.number_input("Deuda de corto plazo (millones)", min_value=0.0,
                                      value=round(med["Deuda_CP_MXN"] / MM, 2), step=0.1)
        with c4:
            st.markdown("**Calidad de cartera e inventario**")
            s_conc = st.number_input("Concentración top 5 clientes (%)", min_value=0.0,
                                     max_value=100.0,
                                     value=round(med["Concentracion_Top5_Clientes_pct"] * 100, 1),
                                     step=1.0)
            s_moro = st.number_input("Morosidad de CxC (%)", min_value=0.0, max_value=100.0,
                                     value=round(med["Morosidad_CxC_pct"] * 100, 1), step=0.5)
            s_obs = st.number_input("Inventario obsoleto (%)", min_value=0.0, max_value=100.0,
                                    value=round(med["Inventario_Obsoleto_pct"] * 100, 1), step=0.5)

        # Variables contables derivadas para mantener consistencia con la base
        ventas = s_ventas * MM
        costo = ventas * s_costo / 100
        cxc = ventas * s_dso / 365
        inv = costo * s_dio / 365
        cxp = costo * s_dpo / 365
        caso = pd.DataFrame([{
            "Fecha": pd.Timestamp(s_fecha),
            "Sector": s_sector,
            "Ventas_12M_MXN": ventas,
            "Crecimiento_Ventas_pct": s_crec / 100,
            "Margen_EBITDA_pct": s_margen / 100,
            "DSO_dias": s_dso, "DIO_dias": s_dio, "DPO_dias": s_dpo,
            "CCC_dias": s_dso + s_dio - s_dpo,
            "CxC_MXN": cxc, "Inventarios_MXN": inv, "CxP_MXN": cxp,
            "NWC_MXN": cxc + inv - cxp,
            "Deuda_CP_MXN": s_deuda * MM,
            "Caja_MXN": s_caja * MM,
            "Linea_Credito_Disponible_MXN": s_linea * MM,
            "Concentracion_Top5_Clientes_pct": s_conc / 100,
            "Morosidad_CxC_pct": s_moro / 100,
            "Inventario_Obsoleto_pct": s_obs / 100,
            "Volatilidad_Ventas_pct": s_vol / 100,
            "Estacionalidad_bin": int(s_est),
            "EBITDA_MXN": ventas * s_margen / 100,
        }])

        prob = float(res["modelo"].predict_proba(caso)[:, 1][0])
        con_tension = prob >= umbral

        st.divider()
        r1, r2 = st.columns([2, 3])
        with r1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": C_TENSION if con_tension else C_SIN_TENSION},
                       "threshold": {"line": {"color": "#333", "width": 3},
                                     "value": umbral * 100}},
                title={"text": "Probabilidad de tensión"}))
            fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig_g, width="stretch")
            if con_tension:
                st.error(f"Clasificación: **con tensión de liquidez** "
                         f"(probabilidad ≥ umbral {umbral:.2f})")
            else:
                st.success(f"Clasificación: **sin tensión de liquidez** "
                           f"(probabilidad < umbral {umbral:.2f})")
        with r2:
            st.markdown("**Variables calculadas a partir de lo capturado**")
            feats = construir_features(caso).iloc[0]
            calc = pd.DataFrame({
                "Variable": ["CxC", "Inventarios", "CxP", "Capital de trabajo neto",
                             "Ciclo de conversión de efectivo", "EBITDA",
                             "Caja / Deuda CP", "(Caja + Línea) / Deuda CP",
                             "NWC / Ventas", "Deuda CP / EBITDA"],
                "Valor": [f"{cxc / MM:,.2f} millones", f"{inv / MM:,.2f} millones",
                          f"{cxp / MM:,.2f} millones", f"{(cxc + inv - cxp) / MM:,.2f} millones",
                          f"{s_dso + s_dio - s_dpo:,.0f} días",
                          f"{ventas * s_margen / 100 / MM:,.2f} millones",
                          f"{feats['Cobertura_Caja_DeudaCP']:.2f}x",
                          f"{feats['Liquidez_Disponible_DeudaCP']:.2f}x",
                          pct(feats["NWC_sobre_Ventas"]),
                          f"{feats['DeudaCP_sobre_EBITDA']:.2f}x"],
            })
            st.dataframe(calc, hide_index=True, width="stretch")
            st.caption("Una razón de 99.00x indica denominador cero o negativo "
                       "(sin deuda CP, o EBITDA ≤ 0).")

    # ---------------------------------------------------------------------------
    # 4. Carga por lotes
    # ---------------------------------------------------------------------------
    with tab_lotes:
        st.subheader("Calificar varios casos")
        st.markdown(
            "Sube un Excel o CSV con las mismas columnas que la base de entrenamiento. "
            "Mes, trimestre y las cuatro razones se calculan automáticamente. "
            "Si el archivo incluye `Tension_Liquidez_bin`, también se evalúa el desempeño.")

        plantilla = res["referencia"][COLUMNAS_REQUERIDAS].head(3)
        buf_pl = io.BytesIO()
        plantilla.to_excel(buf_pl, index=False, sheet_name="Casos")
        st.download_button("Descargar plantilla con 3 casos de ejemplo", buf_pl.getvalue(),
                           file_name="plantilla_casos_tension_liquidez.xlsx")

        lote = st.file_uploader("Archivo a calificar", type=["xlsx", "xls", "csv"], key="lote")
        if lote is not None:
            dl = leer_archivo(lote.getvalue(), lote.name)
            falt = validar_columnas(dl)
            if falt:
                st.error("Faltan estas columnas: " + ", ".join(falt))
            else:
                nulos = dl[COLUMNAS_REQUERIDAS].isna().any(axis=1)
                if nulos.any():
                    st.warning(f"{int(nulos.sum())} filas tienen valores vacíos en columnas "
                               "requeridas y se omitieron.")
                dv = dl.loc[~nulos].copy()
                dv["Prob_Tension_Predicha"] = res["modelo"].predict_proba(
                    dv[COLUMNAS_REQUERIDAS])[:, 1]
                dv["Tension_Predicha"] = (dv["Prob_Tension_Predicha"] >= umbral).astype(int)
                dv["Umbral_Aplicado"] = umbral
                dv = dv.drop(columns=[c for c in COLUMNAS_EXCLUIDAS if c in dv.columns
                                      and c != "ID_Observacion"])

                k1, k2, k3 = st.columns(3)
                k1.metric("Casos calificados", f"{len(dv):,}")
                k2.metric("Alertas de tensión", f"{int(dv['Tension_Predicha'].sum()):,}",
                          pct(dv["Tension_Predicha"].mean()), delta_color="off")
                k3.metric("Probabilidad promedio", pct(dv["Prob_Tension_Predicha"].mean()))

                if TARGET in dv.columns and dv[TARGET].notna().all() and dv[TARGET].nunique() == 2:
                    ml = metricas_en_umbral(dv[TARGET].astype(int),
                                            dv["Prob_Tension_Predicha"].values, umbral)
                    st.markdown(
                        f"**Desempeño contra el valor real:** ROC-AUC "
                        f"{roc_auc_score(dv[TARGET], dv['Prob_Tension_Predicha']):.3f} · "
                        f"Recall {pct(ml['recall'])} · Precisión {pct(ml['precision'])} · "
                        f"F1 {ml['f1']:.3f}")

                orden = dv.sort_values("Prob_Tension_Predicha", ascending=False)
                st.dataframe(orden.style.format({"Prob_Tension_Predicha": "{:.1%}"}),
                             width="stretch", height=380)

                buf_res = io.BytesIO()
                orden.to_excel(buf_res, index=False, sheet_name="Resultados")
                st.download_button("Descargar resultados (.xlsx)", buf_res.getvalue(),
                                   file_name=f"resultados_tension_{date.today():%Y%m%d}.xlsx")

    # ---------------------------------------------------------------------------
    # 5. Datos
    # ---------------------------------------------------------------------------
    with tab_datos:
        ref = res["referencia"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Observaciones", f"{len(ref):,}")
        d2.metric("Entrenamiento / prueba", f"{res['n_train']} / {res['n_test']}")
        d3.metric("Casos con tensión", pct(res["tasa_tension"]))
        d4.metric("Entrenado", res["fecha_entrenamiento"])

        st.markdown("**Variables del modelo**")
        st.markdown(
            f"- Categórica: `Sector` (codificación one-hot)\n"
            f"- Numéricas base ({len(NUMERICAS_BASE)}): " +
            ", ".join(f"`{c}`" for c in NUMERICAS_BASE) + "\n"
            "- Derivadas de fecha: `Mes`, `Trimestre`\n"
            "- Razones: " + ", ".join(f"`{c}`" for c in RAZONES) + "\n"
            "- Excluidas: " + ", ".join(f"`{c}`" for c in COLUMNAS_EXCLUIDAS) +
            ", `Fecha` (solo se usa para derivar mes y trimestre)")

        st.markdown("**Vista previa**")
        st.dataframe(ref.head(50), width="stretch", height=320)


if __name__ == "__main__":
    main()
