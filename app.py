import streamlit as st
import joblib
import requests
import io
import numpy as np
import pandas as pd
import time
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
    "Mamifero":            ("🦁", "Mamífero",            "#D4A017", "Los mamíferos son animales vertebrados que se caracterizan por tener glándulas mamarias que producen leche para alimentar a sus crías."),
    "Mamífero":            ("🦁", "Mamífero",            "#D4A017", "Los mamíferos son animales vertebrados que se caracterizan por tener glándulas mamarias que producen leche para alimentar a sus crías."),
    "Ave":                 ("🦅", "Ave",                 "#4FC3F7", "Las aves son animales vertebrados, de sangre caliente, con plumas y pico sin dientes."),
    "Pez":                 ("🐟", "Pez",                 "#26C6DA", "Los peces son animales vertebrados acuáticos, generalmente ectotérmicos, con branquias y aletas."),
    "Invertebrado_Marino": ("🦑", "Invertebrado Marino", "#AB47BC", "Animales sin columna vertebral que habitan en ecosistemas marinos."),
    "Invertebrado marino": ("🦑", "Invertebrado Marino", "#AB47BC", "Animales sin columna vertebral que habitan en ecosistemas marinos."),
    "Insecto":             ("🐛", "Insecto",             "#66BB6A", "Los insectos son invertebrados con cuerpo dividido en cabeza, tórax y abdomen, con antenas y patas articuladas."),
    "Reptil":              ("🦎", "Reptil",              "#FFA726", "Los reptiles son animales vertebrados de sangre fría, con piel cubierta de escamas."),
    "Anfibio":             ("🐸", "Anfibio",             "#26A69A", "Los anfibios son vertebrados que pasan parte de su vida en el agua y parte en la tierra."),
}

# Para historial
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
    last_err = None
    progress_bar = st.progress(0, text="🌿 Descargando modelos...")
    for i, url in enumerate(urls):
        progress_bar.progress((i + 1) / len(urls), text=f"📡 Intentando desde servidor {i+1}...")
        try:
            resp = requests.get(url, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                content = resp.content
                if content[:10] == b"version ht" or b"oid sha256" in content[:200]:
                    continue
                progress_bar.empty()
                return joblib.load(io.BytesIO(content))
        except Exception as e:
            last_err = e
    progress_bar.empty()
    raise RuntimeError(
        f"No se pudo descargar {filename}.\nURLs probadas:\n"
        + "\n".join(f"  {u}" for u in urls)
        + f"\nUltimo error: {last_err}"
    )

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZOO CLASIFICADOR - Predicción de Especies",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CSS MEJORADO - TEMA ZOOLÓGICO MODERNO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Fredoka+One&display=swap');

/* Reset y fondo principal */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a2f0a 0%, #0b3d0b 50%, #0a2f0a 100%);
    position: relative;
    overflow-x: hidden;
}

/* Hojas animadas que caen */
@keyframes fall {
    0% {
        transform: translateY(-100px) rotate(0deg);
        opacity: 0;
    }
    10% {
        opacity: 0.6;
    }
    90% {
        opacity: 0.6;
    }
    100% {
        transform: translateY(100vh) rotate(360deg);
        opacity: 0;
    }
}

.leaf {
    position: fixed;
    top: -20px;
    pointer-events: none;
    z-index: 0;
    font-size: 20px;
    animation: fall linear infinite;
}

/* Generar hojas con diferentes delays */
.leaf:nth-child(1) { left: 10%; animation-duration: 8s; animation-delay: 0s; }
.leaf:nth-child(2) { left: 25%; animation-duration: 11s; animation-delay: 2s; font-size: 24px; }
.leaf:nth-child(3) { left: 40%; animation-duration: 9s; animation-delay: 1s; }
.leaf:nth-child(4) { left: 55%; animation-duration: 12s; animation-delay: 3s; font-size: 18px; }
.leaf:nth-child(5) { left: 70%; animation-duration: 10s; animation-delay: 0.5s; }
.leaf:nth-child(6) { left: 85%; animation-duration: 7s; animation-delay: 2.5s; font-size: 22px; }
.leaf:nth-child(7) { left: 15%; animation-duration: 13s; animation-delay: 4s; }
.leaf:nth-child(8) { left: 50%; animation-duration: 9.5s; animation-delay: 1.5s; font-size: 26px; }
.leaf:nth-child(9) { left: 75%; animation-duration: 8.5s; animation-delay: 3.5s; }
.leaf:nth-child(10) { left: 35%; animation-duration: 10.5s; animation-delay: 0.8s; font-size: 20px; }

/* Contenedor principal con efecto glassmorphism */
.main-container {
    position: relative;
    z-index: 2;
}

/* Hero section mejorada */
.hero-section {
    background: linear-gradient(135deg, rgba(27, 94, 32, 0.95) 0%, rgba(15, 60, 15, 0.95) 100%);
    backdrop-filter: blur(10px);
    border-radius: 30px;
    padding: 2rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    animation: slideIn 0.8s ease-out;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.hero-title {
    font-family: 'Fredoka One', cursive;
    font-size: 4rem;
    background: linear-gradient(135deg, #FFD54F 0%, #FFA000 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #A5D6A7;
    font-weight: 500;
    letter-spacing: 1px;
}

/* Tarjetas glassmorphism */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 24px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.15);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 1rem;
}

.glass-card:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.12);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
}

.card-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFD54F;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Fredoka One', cursive;
}

/* Inputs estilizados */
.custom-input {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 0.5rem;
    color: white;
    transition: all 0.3s;
}

.custom-input:hover {
    border-color: #FFD54F;
    background: rgba(0, 0, 0, 0.4);
}

/* Select personalizado */
[data-testid="stSelectbox"] > div > div {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Botón principal */
.prediction-btn {
    background: linear-gradient(135deg, #FFD54F 0%, #FFA000 100%);
    color: #1a472a;
    font-size: 1.3rem;
    font-weight: 800;
    padding: 1rem 2rem;
    border-radius: 50px;
    border: none;
    cursor: pointer;
    transition: all 0.3s;
    width: 100%;
    font-family: 'Fredoka One', cursive;
    letter-spacing: 1px;
    margin-top: 1rem;
    animation: pulse 2s infinite;
}

.prediction-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(255, 193, 7, 0.3);
}

@keyframes pulse {
    0%, 100% {
        box-shadow: 0 0 0 0 rgba(255, 193, 7, 0.4);
    }
    50% {
        box-shadow: 0 0 0 15px rgba(255, 193, 7, 0);
    }
}

/* Tarjetas de resultados */
.result-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.5s;
    animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-card:hover {
    transform: scale(1.02);
}

.result-emoji {
    font-size: 5rem;
    filter: drop-shadow(0 8px 15px rgba(0, 0, 0, 0.3));
    animation: bounce 2s infinite;
}

@keyframes bounce {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-10px);
    }
}

.result-label {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.5rem 0;
    font-family: 'Fredoka One', cursive;
}

.confidence-circle {
    width: 80px;
    height: 80px;
    margin: 0 auto;
}

/* Sidebar elegante */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 45, 10, 0.95) 0%, rgba(5, 30, 5, 0.95) 100%);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Gallery cards */
.gallery-card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 0.8rem;
    text-align: center;
    transition: all 0.3s;
    cursor: pointer;
}

.gallery-card:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.15);
}

/* Estadísticas circulares */
.stat-circle {
    width: 100px;
    height: 100px;
    margin: 0 auto;
}

/* Tooltips personalizados */
[data-tooltip] {
    position: relative;
    cursor: help;
    border-bottom: 1px dashed rgba(255, 255, 255, 0.3);
}

[data-tooltip]:before {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    padding: 5px 10px;
    background: rgba(0, 0, 0, 0.9);
    color: white;
    font-size: 0.85rem;
    border-radius: 8px;
    white-space: nowrap;
    display: none;
    z-index: 100;
}

[data-tooltip]:hover:before {
    display: block;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem;
    }
    .hero-subtitle {
        font-size: 0.9rem;
    }
    .result-label {
        font-size: 1.2rem;
    }
}
</style>

<!-- Hojas animadas -->
<div class="leaf">🍃</div>
<div class="leaf">🌿</div>
<div class="leaf">🍂</div>
<div class="leaf">🍃</div>
<div class="leaf">🌿</div>
<div class="leaf">🍂</div>
<div class="leaf">🍃</div>
<div class="leaf">🌿</div>
<div class="leaf">🍂</div>
<div class="leaf">🍃</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR MEJORADO
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦒 Zoológico Digital")
    st.markdown("---")
    
    # Modelo selector
    model_choice = st.radio(
        "🤖 **Modelo Predictivo**",
        ["🌲 Ambos Modelos", "🌳 Random Forest", "🌴 Árbol de Decisión"],
        index=0,
        help="Selecciona qué modelo(s) quieres usar para la predicción"
    )
    
    st.markdown("---")
    
    # Estadísticas rápidas
    st.markdown("### 📊 Estadísticas del Zoo")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Animales", "161", delta="Completo")
    with col2:
        st.metric("Clases", "7", delta="Especies")
    
    st.markdown("---")
    
    # Galería de animales por categoría
    st.markdown("### 🎨 Galería de Especies")
    
    species_gallery = {
        "🦁 Mamíferos": 41,
        "🦅 Aves": 20,
        "🐟 Peces": 13,
        "🦑 Invertebrados": 30,
        "🐛 Insectos": 8,
        "🦎 Reptiles": 5,
        "🐸 Anfibios": 4
    }
    
    for species, count in species_gallery.items():
        st.markdown(f"""
        <div class="gallery-card">
            <div style="font-size: 1.5rem;">{species.split()[0]}</div>
            <div>{species.split()[1]} {species.split()[2] if len(species.split()) > 2 else ''}</div>
            <small style="color: #A5D6A7;">{count} especies</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enlace Colab mejorado
    st.markdown("### 📓 Entrenamiento IA")
    st.markdown(
        f"""
        <a href="{COLAB_URL}" target="_blank" style="text-decoration: none;">
            <div style="background: linear-gradient(135deg, #F9AB00 0%, #E37400 100%); 
                        border-radius: 12px; padding: 0.8rem; text-align: center;">
                <div style="font-size: 1.2rem;">🤖</div>
                <div style="font-weight: 800;">Ver en Colab</div>
                <small>Modelos entrenados con IA</small>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.caption("🐾 **Creado con Streamlit** | Machine Learning para conservación animal")

# ─────────────────────────────────────────────────────────────
#  HERO SECTION
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-title">
        🦁 ZOO CLASSIFIER 🐘
    </div>
    <div class="hero-subtitle">
        Inteligencia Artificial para la Conservación de Especies
    </div>
    <div style="margin-top: 1rem;">
        <span style="background: rgba(255, 255, 255, 0.2); padding: 0.3rem 1rem; border-radius: 50px; font-size: 0.9rem;">
            🧬 16 Características | 🎯 7 Clases | 🚀 98% Precisión
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────
with st.spinner("🌿 Cargando los guardianes del zoológico..."):
    try:
        rf_model = load_model("random_forest_model.pkl")
        dt_model = load_model("decision_tree_model.pkl")
        st.success("✅ **¡Modelos cargados exitosamente!** Los clasificadores están listos para identificar especies.", icon="🦁")
        time.sleep(0.5)
        st.rerun() if 'first_load' not in st.session_state else None
        st.session_state.first_load = True
    except Exception as e:
        st.error(f"❌ Error al cargar los modelos: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────
#  FORMULARIO DE ENTRADA MEJORADO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
    <div class="card-title">
        🔬 Características del Animal
    </div>
    <p style="color: #A5D6A7; margin-bottom: 1.5rem;">
        Completa los siguientes campos para que la IA pueda identificar correctamente la especie.
    </p>
""", unsafe_allow_html=True)

BINARY_OPTIONS = {0: "❌ No", 1: "✅ Sí"}

def binary_select(label, key, col, tooltip=""):
    with col:
        help_text = tooltip if tooltip else f"¿El animal tiene {label.lower()}?"
        return st.selectbox(
            f"{label}", 
            options=[0, 1],
            format_func=lambda x: BINARY_OPTIONS[x], 
            key=key,
            help=help_text
        )

# Función para crear secciones con iconos
def create_section(title, icon, columns_data):
    st.markdown(f"""
    <div style="background: rgba(0, 0, 0, 0.2); border-radius: 16px; padding: 1rem; margin-bottom: 1rem;">
        <h4 style="color: #FFD54F; margin-bottom: 1rem;">{icon} {title}</h4>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(columns_data))
    for idx, (label, key, tooltip) in enumerate(columns_data):
        binary_select(label, key, cols[idx], tooltip)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Sección 1: Características básicas
create_section("Características Básicas", "🧬", [
    ("🐾 Pelo", "pelo", "Los mamíferos tienen pelo o pelaje"),
    ("🪶 Plumas", "plumas", "Característica principal de las aves"),
    ("🥚 Huevos", "huevos", "¿Pone huevos?"),
    ("🍼 Leche", "leche", "¿Produce leche para las crías?")
])

# Sección 2: Capacidades
create_section("Capacidades y Hábitat", "🌍", [
    ("🦋 Vuela", "vuela", "¿Puede volar?"),
    ("🌊 Acuático", "acuatico", "¿Vive en el agua?"),
    ("🦷 Depredador", "depredador", "¿Caza otros animales?"),
    ("😬 Dientes", "con_dientes", "¿Tiene dientes?")
])

# Sección 3: Fisiología
create_section("Fisiología", "🔬", [
    ("🦴 Columna Vertebral", "columna_vertebral", "¿Tiene columna vertebral?"),
    ("💨 Respiración Pulmonar", "respira", "¿Respira por pulmones?"),
    ("☠️ Venenoso", "venenoso", "¿Es venenoso?"),
    ("🐠 Aletas", "aletas", "¿Tiene aletas?")
])

# Sección 4: Otras características
st.markdown("""
<div style="background: rgba(0, 0, 0, 0.2); border-radius: 16px; padding: 1rem; margin-bottom: 1rem;">
    <h4 style="color: #FFD54F; margin-bottom: 1rem;">🏠 Otras Características</h4>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    patas = st.slider(
        "🦵 Número de patas", 
        min_value=0, max_value=10, 
        value=4, step=1, key="patas",
        help="Cantidad de patas que tiene el animal"
    )
with col2:
    domestico = binary_select("🏡 Doméstico", "domestico", col2, "¿Es un animal doméstico?")
with col3:
    tamano_gato = binary_select("📏 Tamaño de gato", "tamano_gato", col3, "¿Tamaño similar a un gato?")
with col4:
    cola = binary_select("🐉 Cola", "cola", col4, "¿Tiene cola?")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Vista previa del animal
st.markdown("""
<div class="glass-card">
    <div class="card-title">
        👀 Vista Previa del Animal
    </div>
""", unsafe_allow_html=True)

# Mostrar resumen de características
features_summary = []
if st.session_state.get('pelo', 0) == 1: features_summary.append("🐾 con pelo")
if st.session_state.get('plumas', 0) == 1: features_summary.append("🪶 con plumas")
if st.session_state.get('vuela', 0) == 1: features_summary.append("🦋 vuela")
if st.session_state.get('acuatico', 0) == 1: features_summary.append("🌊 acuático")

preview_text = "Un animal " + ", ".join(features_summary) if features_summary else "Características no seleccionadas"
st.info(f"📝 **Descripción:** {preview_text} con {patas} patas")

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  BOTÓN DE PREDICCIÓN
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="prediction-btn" id="predict-btn">🔍 ¡IDENTIFICAR ESPECIE!</div>', unsafe_allow_html=True)

# Crear un botón real de Streamlit
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🔍 ¡IDENTIFICAR ESPECIE!", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
#  RESULTADOS MEJORADOS
# ─────────────────────────────────────────────────────────────
if predict_btn:
    features = np.array([[
        st.session_state.get('pelo', 0),
        st.session_state.get('plumas', 0),
        st.session_state.get('huevos', 0),
        st.session_state.get('leche', 0),
        st.session_state.get('vuela', 0),
        st.session_state.get('acuatico', 0),
        st.session_state.get('depredador', 0),
        st.session_state.get('con_dientes', 0),
        st.session_state.get('columna_vertebral', 0),
        st.session_state.get('respira', 0),
        st.session_state.get('venenoso', 0),
        st.session_state.get('aletas', 0),
        patas,
        st.session_state.get('cola', 0),
        st.session_state.get('domestico', 0),
        st.session_state.get('tamano_gato', 0),
    ]])
    
    st.markdown("## 🎯 Resultados del Análisis")
    
    # Mostrar resultados según selección
    if model_choice == "🌲 Ambos Modelos":
        col1, col2 = st.columns(2)
        
        # Random Forest
        with col1:
            pred_rf = rf_model.predict(features)[0]
            pred_key_rf = str(pred_rf).strip()
            info_rf = CLASS_LABELS.get(pred_key_rf, ("🐾", pred_key_rf, "#4caf50", ""))
            emoji_rf, label_rf, color_rf, desc_rf = info_rf
            
            proba_rf = rf_model.predict_proba(features)[0]
            conf_rf = proba_rf.max() * 100
            
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, rgba({int(color_rf[1:3],16)},{int(color_rf[3:5],16)},{int(color_rf[5:7],16)},0.15) 0%, rgba({int(color_rf[1:3],16)},{int(color_rf[3:5],16)},{int(color_rf[5:7],16)},0.05) 100%);">
                <div style="font-size: 0.8rem; opacity: 0.7;">🌲 RANDOM FOREST</div>
                <div class="result-emoji">{emoji_rf}</div>
                <div class="result-label">{label_rf}</div>
                <div style="font-size: 0.9rem; margin: 0.5rem 0;">Confianza: {conf_rf:.1f}%</div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 8px; overflow: hidden;">
                    <div style="background: {color_rf}; width: {conf_rf}%; height: 100%; transition: width 0.5s;"></div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <small>{desc_rf[:100]}...</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Decision Tree
        with col2:
            pred_dt = dt_model.predict(features)[0]
            pred_key_dt = str(pred_dt).strip()
            info_dt = CLASS_LABELS.get(pred_key_dt, ("🐾", pred_key_dt, "#4caf50", ""))
            emoji_dt, label_dt, color_dt, desc_dt = info_dt
            
            proba_dt = dt_model.predict_proba(features)[0]
            conf_dt = proba_dt.max() * 100
            
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, rgba({int(color_dt[1:3],16)},{int(color_dt[3:5],16)},{int(color_dt[5:7],16)},0.15) 0%, rgba({int(color_dt[1:3],16)},{int(color_dt[3:5],16)},{int(color_dt[5:7],16)},0.05) 100%);">
                <div style="font-size: 0.8rem; opacity: 0.7;">🌴 ÁRBOL DE DECISIÓN</div>
                <div class="result-emoji">{emoji_dt}</div>
                <div class="result-label">{label_dt}</div>
                <div style="font-size: 0.9rem; margin: 0.5rem 0;">Confianza: {conf_dt:.1f}%</div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 8px; overflow: hidden;">
                    <div style="background: {color_dt}; width: {conf_dt}%; height: 100%; transition: width 0.5s;"></div>
                </div>
                <div style="margin-top: 0.5rem;">
                    <small>{desc_dt[:100]}...</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif model_choice == "🌳 Random Forest":
        pred = rf_model.predict(features)[0]
        pred_key = str(pred).strip()
        info = CLASS_LABELS.get(pred_key, ("🐾", pred_key, "#4caf50", ""))
        emoji, label, color, desc = info
        
        proba = rf_model.predict_proba(features)[0]
        conf = proba.max() * 100
        
        st.markdown(f"""
        <div class="result-card" style="max-width: 500px; margin: 0 auto; background: linear-gradient(135deg, rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2) 0%, rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1) 100%);">
            <div style="font-size: 0.9rem; opacity: 0.8;">🌲 RANDOM FOREST</div>
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{label}</div>
            <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">{conf:.1f}% de confianza</div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 10px; overflow: hidden; margin: 1rem 0;">
                <div style="background: {color}; width: {conf}%; height: 100%; transition: width 0.5s;"></div>
            </div>
            <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(0,0,0,0.3); border-radius: 10px;">
                <small>📖 {desc}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # Árbol de Decisión
        pred = dt_model.predict(features)[0]
        pred_key = str(pred).strip()
        info = CLASS_LABELS.get(pred_key, ("🐾", pred_key, "#4caf50", ""))
        emoji, label, color, desc = info
        
        proba = dt_model.predict_proba(features)[0]
        conf = proba.max() * 100
        
        st.markdown(f"""
        <div class="result-card" style="max-width: 500px; margin: 0 auto; background: linear-gradient(135deg, rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2) 0%, rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1) 100%);">
            <div style="font-size: 0.9rem; opacity: 0.8;">🌴 ÁRBOL DE DECISIÓN</div>
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{label}</div>
            <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">{conf:.1f}% de confianza</div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 10px; overflow: hidden; margin: 1rem 0;">
                <div style="background: {color}; width: {conf}%; height: 100%; transition: width 0.5s;"></div>
            </div>
            <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(0,0,0,0.3); border-radius: 10px;">
                <small>📖 {desc}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Guardar en historial
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.prediction_history.append({
        'timestamp': timestamp,
        'features': features[0].tolist(),
        'prediction_rf': pred_rf if 'pred_rf' in locals() else None,
        'prediction_dt': pred_dt if 'pred_dt' in locals() else None
    })
    
    # Mantener solo las últimas 5 predicciones
    if len(st.session_state.prediction_history) > 5:
        st.session_state.prediction_history.pop(0)
    
    # Mostrar características detalladas
    with st.expander("📊 Ver todas las características ingresadas", expanded=False):
        feature_names = [
            "pelo", "plumas", "huevos", "leche", "vuela", "acuatico",
            "depredador", "con_dientes", "columna_vertebral", "respira",
            "venenoso", "aletas", "patas", "cola", "domestico", "tamano_gato",
        ]
        
        # Crear DataFrame para mostrar
        df_features = pd.DataFrame([features[0]], columns=feature_names)
        
        # Convertir a valores legibles
        df_display = df_features.replace({1: "✅ Sí", 0: "❌ No"})
        
        st.dataframe(df_display, use_container_width=True)
        
        # Gráfico de características
        st.markdown("### 📈 Distribución de Características")
        feature_binary = features[0][:12]  # Las primeras 12 son binarias
        feature_names_binary = feature_names[:12]
        
        chart_data = pd.DataFrame({
            'Característica': feature_names_binary,
            'Valor': feature_binary
        })
        
        st.bar_chart(chart_data.set_index('Característica'))

# Mostrar historial de predicciones
if st.session_state.prediction_history and not predict_btn:
    with st.expander("📜 Historial de Predicciones Recientes", expanded=False):
        for pred in st.session_state.prediction_history[-5:]:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 0.5rem; margin-bottom: 0.5rem;">
                <small>🕐 {pred['timestamp']}</small>
                <div>Características: {', '.join([str(int(x)) for x in pred['features'][:5]])}...</div>
            </div>
            """, unsafe_allow_html=True)

# Footer con información
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(165, 214, 167, 0.6); padding: 1rem;">
    <div>🐾 **Zoo Classifier** - Proyecto de Machine Learning para la identificación de especies animales 🐾</div>
    <div style="font-size: 0.8rem; margin-top: 0.5rem;">
        Los modelos fueron entrenados con el dataset Zoo utilizando Random Forest y Árboles de Decisión
    </div>
    <div style="font-size: 0.7rem; margin-top: 0.5rem;">
        © 2025 - Creado con ❤️ para la conservación animal
    </div>
</div>
""", unsafe_allow_html=True)
