import streamlit as st
import joblib
import requests
import io
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
GITHUB_USER   = "juliocastrolimas16-boop"
GITHUB_REPO   = "zoo_PA2"
GITHUB_BRANCH = "main"
MODEL_FOLDER  = "model1"
COLAB_URL     = "https://colab.research.google.com/drive/1mzsUZ0WHfJH9Wf70WA4mVKY3dNTVa6K-?usp=sharing"

CLASS_LABELS = {
    "Mamifero":            ("🦁", "Mamífero",            "#D4A017", "Animales vertebrados con pelo y glándulas mamarias"),
    "Ave":                 ("🦅", "Ave",                 "#4FC3F7", "Vertebrados con plumas, pico y alas"),
    "Pez":                 ("🐟", "Pez",                 "#26C6DA", "Vertebrados acuáticos con branquias y aletas"),
    "Invertebrado_Marino": ("🦑", "Invertebrado Marino", "#AB47BC", "Animales marinos sin columna vertebral"),
    "Invertebrado marino": ("🦑", "Invertebrado Marino", "#AB47BC", "Animales marinos sin columna vertebral"),
    "Insecto":             ("🐛", "Insecto",             "#66BB6A", "Invertebrados con 6 patas y cuerpo segmentado"),
    "Reptil":              ("🦎", "Reptil",              "#FFA726", "Vertebrados de sangre fría con escamas"),
    "Anfibio":             ("🐸", "Anfibio",             "#26A69A", "Vertebrados que viven en agua y tierra"),
}

# Inicializar estado
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ─────────────────────────────────────────────────────────────
#  CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    urls = [
        f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{MODEL_FOLDER}/{filename}",
        f"https://media.githubusercontent.com/media/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{MODEL_FOLDER}/{filename}",
        f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/raw/{GITHUB_BRANCH}/{MODEL_FOLDER}/{filename}",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                content = resp.content
                if content[:10] == b"version ht" or b"oid sha256" in content[:200]:
                    continue
                return joblib.load(io.BytesIO(content))
        except Exception:
            continue
    raise RuntimeError(f"No se pudo descargar {filename}")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zoo Classifier",
    page_icon="🐾",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
#  CSS OPTIMIZADO (Sin animaciones pesadas)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Fuentes modernas pero ligeras */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Fondo simple pero atractivo */
.stApp {
    background: linear-gradient(135deg, #0a2f0a 0%, #0b3d0b 50%, #0a2f0a 100%);
}

/* Tarjetas con efecto glass (más ligero) */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 1rem;
    transition: transform 0.2s;
}

.glass-card:hover {
    transform: translateY(-2px);
}

/* Hero section */
.hero {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(27, 94, 32, 0.9), rgba(15, 60, 15, 0.9));
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 215, 0, 0.3);
}

.hero h1 {
    font-size: 3rem;
    background: linear-gradient(135deg, #FFD54F, #FFA000);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero p {
    color: #A5D6A7;
    font-size: 1.1rem;
}

/* Inputs estilizados */
.stSelectbox > div > div, .stNumberInput > div > div {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    color: white !important;
}

.stSelectbox label, .stNumberInput label {
    color: #FFD54F !important;
    font-weight: 600 !important;
}

/* Botón principal */
.stButton > button {
    background: linear-gradient(135deg, #FFD54F, #FFA000);
    color: #1a472a;
    font-weight: 800;
    font-size: 1.2rem;
    padding: 0.75rem 2rem;
    border-radius: 50px;
    border: none;
    width: 100%;
    transition: all 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(255, 193, 7, 0.3);
}

/* Tarjetas de resultados */
.result-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.result-emoji {
    font-size: 4rem;
}

.result-label {
    font-size: 1.5rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 45, 10, 0.95), rgba(5, 30, 5, 0.95));
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    color: #FFD54F;
}

/* Metric cards */
[data-testid="stMetricValue"] {
    color: #FFD54F;
}

/* Títulos */
h1, h2, h3 {
    color: #FFD54F !important;
}

/* Divider */
hr {
    border-color: rgba(255, 255, 255, 0.1);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦒 Zoo Panel")
    st.markdown("---")
    
    model_choice = st.radio(
        "🎯 **Modelo**",
        ["🌲 Ambos", "🌳 Random Forest", "🌴 Árbol Decisión"],
        index=0,
    )
    
    st.markdown("---")
    
    # Estadísticas rápidas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", "161", delta="animales")
    with col2:
        st.metric("Clases", "7", delta="especies")
    
    st.markdown("---")
    
    # Lista de clases
    st.markdown("### 🎨 Clases")
    classes = [
        "🦁 Mamífero", "🦅 Ave", "🐟 Pez",
        "🦑 Invertebrado Marino", "🐛 Insecto",
        "🦎 Reptil", "🐸 Anfibio"
    ]
    for c in classes:
        st.markdown(f"- {c}")
    
    st.markdown("---")
    
    # Enlace Colab
    st.markdown(f'<a href="{COLAB_URL}" target="_blank" style="text-decoration: none;">'
                f'<div style="background: #F9AB00; border-radius: 10px; padding: 0.5rem; text-align: center; color: white; font-weight: bold;">'
                f'📓 Abrir en Colab</div></a>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🦁 Zoo Classifier 🐘</h1>
    <p>Identifica especies animales con Inteligencia Artificial</p>
    <p style="font-size: 0.9rem;">🎯 16 características | 📊 7 clases | 🚀 Alta precisión</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────
with st.spinner("🌿 Cargando modelos..."):
    try:
        rf_model = load_model("random_forest_model.pkl")
        dt_model = load_model("decision_tree_model.pkl")
        st.success("✅ Modelos cargados correctamente")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────
#  FORMULARIO - Optimizado y organizado
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🔬 Características del Animal")

# Función helper para selects binarios
def binary_select(label, key, tooltip=""):
    return st.selectbox(
        label,
        options=[("❌ No", 0), ("✅ Sí", 1)],
        format_func=lambda x: x[0],
        key=key,
        help=tooltip
    )[1]

# Organizar en columnas para mejor flujo
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Básicas**")
    pelo = binary_select("🐾 Pelo", "pelo", "Característico de mamíferos")
    plumas = binary_select("🪶 Plumas", "plumas", "Característico de aves")
    huevos = binary_select("🥚 Huevos", "huevos", "¿Pone huevos?")
    leche = binary_select("🍼 Leche", "leche", "¿Produce leche?")

with col2:
    st.markdown("**Capacidades**")
    vuela = binary_select("🦋 Vuela", "vuela", "¿Puede volar?")
    acuatico = binary_select("🌊 Acuático", "acuatico", "¿Vive en agua?")
    depredador = binary_select("🦷 Depredador", "depredador", "¿Caza otros animales?")
    con_dientes = binary_select("😬 Dientes", "con_dientes", "¿Tiene dientes?")

with col3:
    st.markdown("**Fisiología**")
    columna = binary_select("🦴 Columna", "columna_vertebral", "¿Tiene columna vertebral?")
    respira = binary_select("💨 Pulmones", "respira", "¿Respira por pulmones?")
    venenoso = binary_select("☠️ Venenoso", "venenoso", "¿Es venenoso?")
    aletas = binary_select("🐠 Aletas", "aletas", "¿Tiene aletas?")

with col4:
    st.markdown("**Otras**")
    cola = binary_select("🐉 Cola", "cola", "¿Tiene cola?")
    domestico = binary_select("🏡 Doméstico", "domestico", "¿Es doméstico?")
    tamano_gato = binary_select("📏 Tamaño gato", "tamano_gato", "¿Tamaño similar a un gato?")
    
    patas = st.number_input(
        "🦵 Patas",
        min_value=0, max_value=10,
        value=4, step=1,
        help="Número de patas"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  BOTÓN PREDECIR
# ─────────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict = st.button("🔍 IDENTIFICAR ESPECIE", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
#  RESULTADOS
# ─────────────────────────────────────────────────────────────
if predict:
    # Construir features
    features = np.array([[
        pelo, plumas, huevos, leche, vuela, acuatico,
        depredador, con_dientes, columna, respira,
        venenoso, aletas, patas, cola, domestico, tamano_gato
    ]])
    
    st.markdown("## 🎯 Resultados")
    
    # Mostrar según selección
    if model_choice == "🌲 Ambos":
        col_rf, col_dt = st.columns(2)
        
        with col_rf:
            pred = rf_model.predict(features)[0]
            pred_key = str(pred).strip()
            emoji, label, color, desc = CLASS_LABELS.get(pred_key, ("🐾", pred_key, "#4caf50", ""))
            proba = rf_model.predict_proba(features)[0]
            conf = proba.max() * 100
            
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size: 0.8rem; opacity: 0.7;">🌲 RANDOM FOREST</div>
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{label}</div>
                <div style="font-size: 1.2rem; font-weight: 600;">{conf:.1f}%</div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 5px; height: 6px; margin: 0.5rem 0;">
                    <div style="background: {color}; width: {conf}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col_dt:
            pred = dt_model.predict(features)[0]
            pred_key = str(pred).strip()
            emoji, label, color, desc = CLASS_LABELS.get(pred_key, ("🐾", pred_key, "#4caf50", ""))
            proba = dt_model.predict_proba(features)[0]
            conf = proba.max() * 100
            
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size: 0.8rem; opacity: 0.7;">🌴 ÁRBOL DECISIÓN</div>
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{label}</div>
                <div style="font-size: 1.2rem; font-weight: 600;">{conf:.1f}%</div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 5px; height: 6px; margin: 0.5rem 0;">
                    <div style="background: {color}; width: {conf}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
    
    elif model_choice == "🌳 Random Forest":
        pred = rf_model.predict(features)[0]
        pred_key = str(pred).strip()
        emoji, label, color, desc = CLASS_LABELS.get(pred_key, ("🐾", pred_key, "#4caf50", ""))
        proba = rf_model.predict_proba(features)[0]
        conf = proba.max() * 100
        
        st.markdown(f"""
        <div class="result-card" style="max-width: 500px; margin: 0 auto;">
            <div style="font-size: 0.8rem; opacity: 0.7;">🌲 RANDOM FOREST</div>
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{label}</div>
            <div style="font-size: 1.5rem; font-weight: 600;">{conf:.1f}%</div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 5px; height: 8px; margin: 1rem 0;">
                <div style="background: {color}; width: {conf}%; height: 100%; border-radius: 5px;"></div>
            </div>
            <small>{desc}</small>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # Árbol Decisión
        pred = dt_model.predict(features)[0]
        pred_key = str(pred).strip()
        emoji, label, color, desc = CLASS_LABELS.get(pred_key, ("🐾", pred_key, "#4caf50", ""))
        proba = dt_model.predict_proba(features)[0]
        conf = proba.max() * 100
        
        st.markdown(f"""
        <div class="result-card" style="max-width: 500px; margin: 0 auto;">
            <div style="font-size: 0.8rem; opacity: 0.7;">🌴 ÁRBOL DECISIÓN</div>
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{label}</div>
            <div style="font-size: 1.5rem; font-weight: 600;">{conf:.1f}%</div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 5px; height: 8px; margin: 1rem 0;">
                <div style="background: {color}; width: {conf}%; height: 100%; border-radius: 5px;"></div>
            </div>
            <small>{desc}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Guardar en historial
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.prediction_history.append({
        'timestamp': timestamp,
        'patas': int(patas)
    })
    if len(st.session_state.prediction_history) > 5:
        st.session_state.prediction_history.pop(0)
    
    # Mostrar características
    with st.expander("📊 Ver características detalladas"):
        feature_names = [
            "Pelo", "Plumas", "Huevos", "Leche", "Vuela", "Acuático",
            "Depredador", "Dientes", "Columna vertebral", "Respira pulmón",
            "Venenoso", "Aletas", "Patas", "Cola", "Doméstico", "Tamaño gato"
        ]
        
        df = pd.DataFrame([{
            name: "✅ Sí" if val == 1 else "❌ No" if val in [0,1] else str(val)
            for name, val in zip(feature_names, features[0])
        }])
        
        st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(165, 214, 167, 0.6); padding: 1rem;">
    <small>🐾 Zoo Classifier - Proyecto de Machine Learning para conservación animal</small>
</div>
""", unsafe_allow_html=True)
