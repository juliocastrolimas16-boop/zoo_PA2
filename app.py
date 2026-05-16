import streamlit as st
import joblib
import requests
import io
import numpy as np

# ─────────────────────────────────────────────
#  CONFIGURACIÓN — cambia estas URLs a las tuyas
# ─────────────────────────────────────────────
GITHUB_USER = "juliocastrolimas16-boop"
GITHUB_REPO = "zoo_PA2"
GITHUB_BRANCH = "main"
MODEL_FOLDER = "model1"

# ─────────────────────────────────────────────
#  Mapa de clases
# ─────────────────────────────────────────────
CLASS_LABELS = {
    "Mamifero":            "🦁 Mamífero",
    "Mamífero":            "🦁 Mamífero",
    "Ave":                 "🦅 Ave",
    "Pez":                 "🐟 Pez",
    "Invertebrado_Marino": "🦑 Invertebrado marino",
    "Invertebrado marino": "🦑 Invertebrado marino",
    "Insecto":             "🐛 Insecto",
    "Reptil":              "🦎 Reptil",
    "Anfibio":             "🐸 Anfibio",
}

# ─────────────────────────────────────────────
#  Carga de modelos (con caché)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    """
    Intenta cargar el modelo probando tres URLs en orden:
      1. raw.githubusercontent.com  (archivos normales)
      2. media.githubusercontent.com (archivos Git LFS)
      3. github.com/raw             (fallback LFS redirect)
    """
    urls = [
        f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{MODEL_FOLDER}/{filename}",
        f"https://media.githubusercontent.com/media/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{MODEL_FOLDER}/{filename}",
        f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/raw/{GITHUB_BRANCH}/{MODEL_FOLDER}/{filename}",
    ]
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                content = resp.content
                # Detectar puntero LFS (texto plano ~130 bytes, no binario)
                if content[:10] == b"version ht" or b"oid sha256" in content[:200]:
                    continue  # no es el binario real, probar siguiente URL
                return joblib.load(io.BytesIO(content))
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"No se pudo descargar **{filename}**.\n\nURLs probadas:\n"
        + "\n".join(f"- `{u}`" for u in urls)
        + f"\n\nÚltimo error: {last_err}"
    )

# ─────────────────────────────────────────────
#  Página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Animales",
    page_icon="🐾",
    layout="wide",
)

# ── Estilos ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo degradado */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Header central */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 800;
    background: linear-gradient(90deg, #f9d423, #ff4e50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 .4rem;
}
.hero p {
    color: #c4b8f0;
    font-size: 1.05rem;
    margin: 0;
}

/* Tarjeta contenedor */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.4rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f9d423;
    margin-bottom: 1rem;
    letter-spacing: .04em;
    text-transform: uppercase;
}

/* Etiquetas de inputs */
label[data-testid="stWidgetLabel"] > div {
    color: #e2daf8 !important;
    font-size: .9rem !important;
}

/* Selectbox / número */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.08) !important;
    color: #fff !important;
    border-color: rgba(255,255,255,0.18) !important;
    border-radius: 8px !important;
}

/* Botón principal */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #f9d423, #ff4e50);
    color: #1a1a2e;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 12px;
    padding: .85rem 1.5rem;
    cursor: pointer;
    transition: transform .15s, box-shadow .15s;
    letter-spacing: .03em;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(249,212,35,.35);
}

/* Resultado */
.result-box {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-box .model-name {
    font-size: .85rem;
    font-weight: 500;
    color: #c4b8f0;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .5rem;
}
.result-box .animal {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
}
.result-box .prob {
    font-size: .9rem;
    color: #a09acf;
    margin-top: .3rem;
}
.rf-box  { background: linear-gradient(135deg, rgba(249,212,35,.15), rgba(249,212,35,.04)); border: 1px solid rgba(249,212,35,.3); }
.dt-box  { background: linear-gradient(135deg, rgba(255,78,80,.15),  rgba(255,78,80,.04));  border: 1px solid rgba(255,78,80,.3); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(20, 18, 50, 0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #d4cdf5 !important; }

/* Divisor */
hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🐾 Clasificador de Animales</h1>
  <p>Ingresa las características del animal y obtén su clasificación con dos modelos de Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar — selector de modelo ──────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    model_choice = st.radio(
        "Seleccionar modelo",
        ["Ambos modelos", "Random Forest", "Árbol de Decisión"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Clases disponibles**")
    seen = set()
    for v in CLASS_LABELS.values():
        if v not in seen:
            st.markdown(f"- {v}")
            seen.add(v)

# ── Cargar modelos ────────────────────────────
with st.spinner("Cargando modelos desde GitHub…"):
    try:
        rf_model = load_model("random_forest_model.pkl")
        dt_model = load_model("decision_tree_model.pkl")
        st.success("✅ Modelos cargados correctamente", icon="✅")
    except Exception as e:
        st.error(f"❌ Error al cargar los modelos: {e}")
        st.info(
            "Verifica que `GITHUB_USER`, `GITHUB_REPO` y `GITHUB_BRANCH` "
            "estén configurados correctamente al inicio de **app.py**."
        )
        st.stop()

# ── Formulario de entrada ─────────────────────
st.markdown("### 📋 Características del animal")

# Columnas de layout
col1, col2, col3 = st.columns(3)

BINARY_OPTIONS = {0: "No (0)", 1: "Sí (1)"}

def binary_select(label, key, col):
    with col:
        val = st.selectbox(label, options=[0, 1],
                           format_func=lambda x: BINARY_OPTIONS[x],
                           key=key)
    return val

# ── Bloque 1: Cobertura corporal ──────────────
st.markdown('<div class="card"><div class="card-title">🧬 Cobertura corporal & reproducción</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
pelo              = binary_select("Pelo",              "pelo",    c1)
plumas            = binary_select("Plumas",            "plumas",  c2)
huevos            = binary_select("Huevos",            "huevos",  c3)
leche             = binary_select("Leche",             "leche",   c4)
st.markdown('</div>', unsafe_allow_html=True)

# ── Bloque 2: Capacidades ─────────────────────
st.markdown('<div class="card"><div class="card-title">🌍 Capacidades & hábitat</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
vuela             = binary_select("Vuela",             "vuela",     c1)
acuatico          = binary_select("Acuático",          "acuatico",  c2)
depredador        = binary_select("Depredador",        "depredador",c3)
con_dientes       = binary_select("Con dientes",       "con_dientes",c4)
st.markdown('</div>', unsafe_allow_html=True)

# ── Bloque 3: Fisiología ──────────────────────
st.markdown('<div class="card"><div class="card-title">🔬 Fisiología & morfología</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
columna_vertebral = binary_select("Columna vertebral","columna_vertebral", c1)
respira           = binary_select("Respira (pulmones)","respira",          c2)
venenoso          = binary_select("Venenoso",          "venenoso",         c3)
aletas            = binary_select("Aletas",            "aletas",           c4)
cola              = binary_select("Cola",              "cola",             c5)
st.markdown('</div>', unsafe_allow_html=True)

# ── Bloque 4: Otras características ──────────
st.markdown('<div class="card"><div class="card-title">🏠 Otras características</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    patas = st.number_input("Número de patas", min_value=0, max_value=10,
                            value=4, step=1, key="patas")
domestico         = binary_select("Doméstico",         "domestico",   c2)
tamano_gato       = binary_select("Tamaño de gato",    "tamano_gato", c3)
st.markdown('</div>', unsafe_allow_html=True)

# ── Botón de predicción ───────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Obtener predicción", use_container_width=True)

# ── Resultado ─────────────────────────────────
if predict_btn:
    features = np.array([[
        pelo, plumas, huevos, leche, vuela, acuatico,
        depredador, con_dientes, columna_vertebral, respira,
        venenoso, aletas, patas, cola, domestico, tamano_gato,
    ]])

    st.markdown("### 🎯 Resultados de clasificación")
    res_col1, res_col2 = st.columns(2)

    def show_result(model, model_name, box_class, col):
        pred = model.predict(features)[0]          # puede ser str o int
        pred_key = str(pred).strip()
        label = CLASS_LABELS.get(pred_key, f"🐾 {pred_key}")
        prob_str = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            conf = proba.max() * 100
            prob_str = f"<div class='prob'>Confianza: <b>{conf:.1f}%</b></div>"
        with col:
            st.markdown(f"""
            <div class="result-box {box_class}">
              <div class="model-name">{model_name}</div>
              <div class="animal">{label}</div>
              {prob_str}
            </div>
            """, unsafe_allow_html=True)

    show_result(rf_model, "Random Forest",       "rf-box", res_col1)
    show_result(dt_model, "Árbol de Decisión",   "dt-box", res_col2)

    # Resumen de características ingresadas
    with st.expander("📊 Ver características ingresadas"):
        feature_names = [
            "pelo","plumas","huevos","leche","vuela","acuatico",
            "depredador","con_dientes","columna_vertebral","respira",
            "venenoso","aletas","patas","cola","domestico","tamano_gato",
        ]
        feature_values = features[0].astype(int).tolist()
        feat_data = {name: [val] for name, val in zip(feature_names, feature_values)}
        import pandas as pd
        df = pd.DataFrame(feat_data)
        st.dataframe(df, use_container_width=True)
