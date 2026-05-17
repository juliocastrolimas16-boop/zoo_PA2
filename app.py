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
    "Mamifero":            ("🦁", "Mamífero",            "#D4A017", "Vertebrados con pelo y glándulas mamarias"),
    "Mamífero":            ("🦁", "Mamífero",            "#D4A017", "Vertebrados con pelo y glándulas mamarias"),
    "Ave":                 ("🦅", "Ave",                 "#4FC3F7", "Vertebrados con plumas, pico y alas"),
    "Pez":                 ("🐟", "Pez",                 "#26C6DA", "Vertebrados acuáticos con branquias y aletas"),
    "Invertebrado_Marino": ("🦑", "Invertebrado Marino", "#AB47BC", "Animales marinos sin columna vertebral"),
    "Invertebrado marino": ("🦑", "Invertebrado Marino", "#AB47BC", "Animales marinos sin columna vertebral"),
    "Insecto":             ("🐛", "Insecto",             "#66BB6A", "Invertebrados con 6 patas y cuerpo segmentado"),
    "Reptil":              ("🦎", "Reptil",              "#FFA726", "Vertebrados de sangre fría con escamas"),
    "Anfibio":             ("🐸", "Anfibio",             "#26A69A", "Vertebrados que viven en agua y tierra"),
}

if "prediction_history" not in st.session_state:
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
            resp = requests.get(url, timeout=30, allow_redirects=True)
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
    page_title="Zoo Classifier 🐾",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CSS — SELVA + GLASS CARDS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Boogaloo&family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

/* ── Fondo selva ── */
[data-testid="stAppViewContainer"] {
    background-color: #071a07;
    background-image:
        radial-gradient(ellipse at 8%  15%, rgba(27,94,32,0.65)  0%, transparent 50%),
        radial-gradient(ellipse at 92% 85%, rgba(20,70,20,0.60)  0%, transparent 50%),
        radial-gradient(ellipse at 80% 10%, rgba(30,80,10,0.45)  0%, transparent 45%),
        radial-gradient(ellipse at 15% 90%, rgba(15,60,15,0.45)  0%, transparent 45%);
    min-height: 100vh;
}
[data-testid="stAppViewContainer"]::before {
    content: "🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾";
    position: fixed; top:0; left:0; right:0; bottom:0;
    font-size: 3rem; line-height: 5rem; word-spacing: 2.5rem;
    opacity: 0.03; pointer-events: none; z-index: 0;
    overflow: hidden; padding: 1.5rem;
}

/* ── Hero ── */
.zoo-hero {
    background: linear-gradient(135deg, rgba(20,70,20,0.95) 0%, rgba(10,50,10,0.98) 100%);
    border: 1px solid rgba(255,215,0,0.25);
    border-radius: 24px;
    padding: 2.4rem 2rem 1.8rem;
    text-align: center;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,215,0,0.1);
    position: relative;
    overflow: hidden;
}
.zoo-hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(255,215,0,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.zoo-hero h1 {
    font-family: 'Boogaloo', cursive;
    font-size: clamp(2.6rem, 5.5vw, 4.2rem);
    background: linear-gradient(135deg, #FFD54F 20%, #FFA000 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    margin: 0 0 .4rem;
    letter-spacing: .04em;
    filter: drop-shadow(0 4px 12px rgba(255,193,7,0.3));
}
.zoo-hero p { color: #a5d6a7; font-size: 1.05rem; margin: .3rem 0 0; font-weight: 600; }
.zoo-hero .sub { color: #81c784; font-size: .9rem; margin-top: .4rem; }

/* ── Valla madera ── */
.wood-rail {
    width: 100%; height: 8px;
    background: repeating-linear-gradient(90deg,
        #7B5314 0px, #7B5314 26px, #9B7A1A 26px, #9B7A1A 28px);
    border-radius: 4px; opacity: .6;
    box-shadow: 0 3px 8px rgba(0,0,0,.5);
}
.fence-row {
    display: flex; justify-content: space-between;
    padding: 0 1%; margin: 0;
}
.fence-post {
    width: 15px;
    background: linear-gradient(180deg, #9B7A1A 0%, #6B4F10 65%, #4a350a 100%);
    border-radius: 4px 4px 0 0;
    box-shadow: 2px 0 4px rgba(0,0,0,.4);
}
.wood-divider {
    width:100%; height:5px;
    background: repeating-linear-gradient(90deg,
        #5a3e0c 0px, #5a3e0c 22px, #7B5314 22px, #7B5314 24px);
    border-radius:3px; margin:1.5rem 0; opacity:.45;
}

/* ── Glass cards de sección ── */
.zoo-card {
    background: rgba(10,45,10,0.80);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(100,200,100,0.15);
    border-radius: 20px;
    padding: 1.6rem 1.8rem 1.4rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 6px 30px rgba(0,0,0,.45), inset 0 1px 0 rgba(165,214,167,.06);
    position: relative;
    transition: transform .2s;
}
.zoo-card:hover { transform: translateY(-2px); }
.zoo-card::after {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    border-radius:20px 20px 0 0;
    background: linear-gradient(90deg, transparent 0%, rgba(165,214,167,.45) 40%, rgba(255,215,0,.3) 60%, transparent 100%);
}
.zoo-card-title {
    font-family: 'Boogaloo', cursive;
    font-size: 1.25rem; color: #a5d6a7;
    letter-spacing: .06em; margin-bottom: 1rem;
    display: flex; align-items: center; gap: .45rem;
}
.zoo-card-title::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg, rgba(100,200,100,.22), transparent);
    margin-left:.5rem;
}

/* ── Inputs ── */
label[data-testid="stWidgetLabel"] > div {
    color: #FFD54F !important; font-weight: 700 !important; font-size: .86rem !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(0,40,0,0.65) !important; color: #e8f5e9 !important;
    border: 1px solid rgba(100,200,100,.22) !important; border-radius: 10px !important;
}
[data-testid="stNumberInput"] input {
    background: rgba(0,40,0,0.65) !important; color: #e8f5e9 !important;
    border: 1px solid rgba(100,200,100,.22) !important; border-radius: 10px !important;
}

/* ── Botón predecir ── */
div.stButton > button {
    background: linear-gradient(135deg, #FFD54F 0%, #FFA000 100%);
    color: #1a3a00;
    font-family: 'Boogaloo', cursive;
    font-size: 1.35rem; letter-spacing: .08em;
    font-weight: 400;
    border: none; border-radius: 50px;
    padding: .9rem 2rem; width: 100%;
    box-shadow: 0 6px 24px rgba(255,193,7,.35);
    transition: all .22s ease;
    text-shadow: none;
}
div.stButton > button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 12px 36px rgba(255,193,7,.55);
}

/* ── Tarjetas resultado ── */
.result-card {
    border-radius: 22px; padding: 2rem 1.5rem;
    text-align: center; position: relative; overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,.5);
    transition: transform .2s;
}
.result-card:hover { transform: translateY(-4px); }
.result-card::before {
    content:''; position:absolute; top:-50px; right:-50px;
    width:150px; height:150px; border-radius:50%;
    background:rgba(255,255,255,.05); pointer-events:none;
}
.result-model { font-size:.75rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; opacity:.7; margin-bottom:.6rem; }
.result-emoji { font-size:4rem; line-height:1; margin-bottom:.45rem; filter:drop-shadow(0 6px 12px rgba(0,0,0,.6)); }
.result-label { font-family:'Boogaloo',cursive; font-size:2.1rem; color:#fff; text-shadow:0 3px 8px rgba(0,0,0,.5); }
.result-conf  { font-size:1.15rem; font-weight:700; margin:.4rem 0 .2rem; }
.result-desc  { font-size:.82rem; opacity:.7; margin-top:.5rem; }
.bar-bg  { background:rgba(255,255,255,.18); border-radius:5px; height:7px; margin:.6rem auto; width:78%; }
.bar-fill{ height:100%; border-radius:5px; }

/* ── Colab badge ── */
.colab-btn {
    display: flex; align-items: center; justify-content: center; gap: .55rem;
    background: linear-gradient(135deg, #F9AB00 0%, #E37400 100%);
    color: #fff !important; text-decoration: none !important;
    font-family: 'Nunito', sans-serif; font-weight: 800; font-size: .95rem;
    padding: .65rem 1.2rem; border-radius: 14px;
    box-shadow: 0 4px 20px rgba(249,171,0,.4);
    transition: all .2s; margin-bottom: .5rem;
}
.colab-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(249,171,0,.6);
}

/* ── Clase chip ── */
.clase-chip {
    display:flex; align-items:center; gap:.4rem;
    background:rgba(40,110,40,.12);
    border:1px solid rgba(100,200,100,.12);
    border-left:3px solid rgba(100,200,100,.4);
    border-radius:0 8px 8px 0;
    padding:.3rem .75rem; margin-bottom:.26rem;
    font-size:.87rem; color:#b9deba;
}

/* ── Métricas ── */
[data-testid="stMetricValue"] { color: #FFD54F !important; font-family:'Boogaloo',cursive !important; }
[data-testid="stMetricLabel"] { color: #81c784 !important; }
[data-testid="stMetricDelta"] { color: #a5d6a7 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061506 0%, #0a280a 50%, #061506 100%) !important;
    border-right: 1px solid rgba(100,200,100,.1) !important;
}
[data-testid="stSidebar"] * { color: #b9deba !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFD54F !important; }

/* ── Historial ── */
.hist-item {
    background: rgba(20,70,20,.4); border:1px solid rgba(100,200,100,.12);
    border-radius: 10px; padding:.4rem .8rem; margin-bottom:.25rem;
    font-size:.82rem; display:flex; align-items:center; gap:.5rem;
}

/* ── Misc ── */
hr { border-color: rgba(255,255,255,0.08) !important; }
[data-testid="stAlert"] { border-radius: 14px !important; }
[data-testid="stExpander"] {
    background: rgba(10,40,10,0.55) !important;
    border: 1px solid rgba(100,200,100,.12) !important;
    border-radius: 14px !important;
}
h1,h2,h3 { color: #FFD54F !important; }

/* ── Hierba footer ── */
.grass-row { display:flex; justify-content:center; align-items:flex-end; gap:3px; height:60px; overflow:hidden; margin-top:1.8rem; }
.blade { border-radius:4px 4px 0 0; animation:sway 2.2s ease-in-out infinite alternate; }
@keyframes sway {
    0%   { transform:rotate(-10deg); transform-origin:bottom center; }
    100% { transform:rotate( 10deg); transform-origin:bottom center; }
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
        "🎯 Modelo",
        ["🌲 Ambos modelos", "🌳 Random Forest", "🌴 Árbol de Decisión"],
        index=0,
    )

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Animales", "161")
    with c2:
        st.metric("Clases", "7")

    st.markdown("---")
    st.markdown("### 🎨 Especies")
    clases = [
        ("🦁","Mamífero"),("🦅","Ave"),("🐟","Pez"),
        ("🦑","Invertebrado"),("🐛","Insecto"),
        ("🦎","Reptil"),("🐸","Anfibio"),
    ]
    for emoji, nombre in clases:
        st.markdown(f'<div class="clase-chip">{emoji} {nombre}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📓 Notebook")

    # Badge oficial de Colab con estilo propio
    st.markdown(f"""
    <a class="colab-btn" href="{COLAB_URL}" target="_blank">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" height="20"/>
        &nbsp;Abrir en Colab
    </a>
    """, unsafe_allow_html=True)
    st.caption("Entrenamiento completo · Random Forest & Árbol de Decisión · Dataset Zoo")

    # Historial de predicciones
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 🕐 Últimas predicciones")
        for item in reversed(st.session_state.prediction_history):
            st.markdown(
                f'<div class="hist-item">'
                f'{item["emoji"]} <b>{item["label"]}</b>'
                f'<span style="margin-left:auto;opacity:.55">{item["time"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="zoo-hero">
  <h1>🦁 Zoo Classifier 🐘</h1>
  <p>Identifica la especie de cualquier animal usando Inteligencia Artificial</p>
  <p class="sub">🎯 16 características &nbsp;|&nbsp; 📊 7 clases &nbsp;|&nbsp; 🌲 Random Forest &nbsp;+&nbsp; 🌴 Árbol de Decisión</p>
</div>
""", unsafe_allow_html=True)

# Botón Colab centrado bajo el hero
_, col_colab, _ = st.columns([2, 1, 2])
with col_colab:
    st.link_button("📓 Ver Notebook en Colab", COLAB_URL, use_container_width=True)

# Valla decorativa
posts = 54
ph = [40,46,38,52,36,44,42,50,38,48,42,54,36,46,40] * 4
posts_html = "".join(
    f'<div class="fence-post" style="height:{ph[i%len(ph)]}px;margin-top:{54-ph[i%len(ph)]}px"></div>'
    for i in range(posts)
)
st.markdown(f"""
<div style="padding:.4rem 1rem 0;overflow:hidden;">
  <div class="wood-rail"></div>
  <div class="fence-row">{posts_html}</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────
with st.spinner("🌿 Cargando modelos desde GitHub…"):
    try:
        rf_model = load_model("random_forest_model.pkl")
        dt_model = load_model("decision_tree_model.pkl")
        st.success("✅  ¡Modelos listos! Los guardianes del zoo están en posición.", icon="🦺")
    except Exception as e:
        st.error(f"❌ Error al cargar los modelos: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────
#  FORMULARIO
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="wood-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<h3 style="font-family:Boogaloo,cursive;color:#a5d6a7;letter-spacing:.06em;margin-bottom:.3rem;">'
    '🔬 Características del animal</h3>',
    unsafe_allow_html=True,
)

def binary_select(label, key, tooltip=""):
    return st.selectbox(
        label, options=[("❌  No", 0), ("✅  Sí", 1)],
        format_func=lambda x: x[0], key=key, help=tooltip,
    )[1]

# ── Bloque 1 ──
st.markdown('<div class="zoo-card"><div class="zoo-card-title">🧬 Cobertura corporal & reproducción</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: pelo   = binary_select("🐾 Pelo",   "pelo",   "Característico de mamíferos")
with c2: plumas = binary_select("🪶 Plumas", "plumas", "Característico de aves")
with c3: huevos = binary_select("🥚 Huevos", "huevos", "¿Pone huevos?")
with c4: leche  = binary_select("🍼 Leche",  "leche",  "¿Produce leche?")
st.markdown('</div>', unsafe_allow_html=True)

# ── Bloque 2 ──
st.markdown('<div class="zoo-card"><div class="zoo-card-title">🌍 Capacidades & hábitat</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: vuela       = binary_select("🦋 Vuela",        "vuela",       "¿Puede volar?")
with c2: acuatico    = binary_select("🌊 Acuático",     "acuatico",    "¿Vive en agua?")
with c3: depredador  = binary_select("🦷 Depredador",   "depredador",  "¿Caza otros animales?")
with c4: con_dientes = binary_select("😬 Con dientes",  "con_dientes", "¿Tiene dientes?")
st.markdown('</div>', unsafe_allow_html=True)

# ── Bloque 3 ──
st.markdown('<div class="zoo-card"><div class="zoo-card-title">🔬 Fisiología & morfología</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
with c1: columna  = binary_select("🦴 Columna vert.",    "columna_vertebral", "¿Tiene columna vertebral?")
with c2: respira  = binary_select("💨 Respira (pulmón)", "respira",           "¿Respira por pulmones?")
with c3: venenoso = binary_select("☠️ Venenoso",         "venenoso",          "¿Es venenoso?")
with c4: aletas   = binary_select("🐠 Aletas",           "aletas",            "¿Tiene aletas?")
with c5: cola     = binary_select("🐉 Cola",             "cola",              "¿Tiene cola?")
st.markdown('</div>', unsafe_allow_html=True)

# ── Bloque 4 ──
st.markdown('<div class="zoo-card"><div class="zoo-card-title">🏠 Otras características</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    patas = st.number_input("🦵 Número de patas", min_value=0, max_value=10,
                            value=4, step=1, help="Número total de patas")
with c2: domestico   = binary_select("🏡 Doméstico",      "domestico",   "¿Es animal doméstico?")
with c3: tamano_gato = binary_select("📏 Tamaño de gato", "tamano_gato", "¿Tiene un tamaño similar al de un gato?")
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  BOTÓN PREDECIR
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="wood-divider"></div>', unsafe_allow_html=True)
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    predict = st.button("🔍  ¡IDENTIFICAR ESPECIE!", use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  RESULTADO
# ─────────────────────────────────────────────────────────────
if predict:
    features = np.array([[
        pelo, plumas, huevos, leche, vuela, acuatico,
        depredador, con_dientes, columna, respira,
        venenoso, aletas, patas, cola, domestico, tamano_gato,
    ]])

    st.markdown(
        '<h3 style="font-family:Boogaloo,cursive;color:#a5d6a7;letter-spacing:.06em;margin-top:1.2rem;">'
        '🎯 Resultados del análisis</h3>',
        unsafe_allow_html=True,
    )

    def render_result(model, model_name, col):
        pred     = model.predict(features)[0]
        pred_key = str(pred).strip()
        emoji, label, color, desc = CLASS_LABELS.get(pred_key, ("🐾", pred_key, "#4caf50", ""))
        conf = 0.0
        bar_html = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            conf  = proba.max() * 100
            bar_html = f"""
            <div class="bar-bg">
              <div class="bar-fill" style="width:{int(conf)}%;background:{color};"></div>
            </div>"""
        with col:
            st.markdown(f"""
            <div class="result-card" style="
                background: linear-gradient(140deg, rgba(10,40,10,0.95) 0%, rgba(5,25,5,0.98) 100%);
                border: 2px solid {color}40;">
              <div class="result-model">{model_name}</div>
              <div class="result-emoji">{emoji}</div>
              <div class="result-label">{label}</div>
              <div class="result-conf" style="color:{color};">{conf:.1f}% confianza</div>
              {bar_html}
              <div class="result-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        return emoji, label, conf

    if model_choice == "🌲 Ambos modelos":
        col_rf, _, col_dt = st.columns([1, .06, 1])
        e1, l1, _ = render_result(rf_model, "🌲 RANDOM FOREST",     col_rf)
        e2, l2, _ = render_result(dt_model, "🌴 ÁRBOL DE DECISIÓN", col_dt)
        last_emoji, last_label = e1, l1

    elif model_choice == "🌳 Random Forest":
        _, col_c, _ = st.columns([1, 2, 1])
        last_emoji, last_label, _ = render_result(rf_model, "🌲 RANDOM FOREST", col_c)

    else:
        _, col_c, _ = st.columns([1, 2, 1])
        last_emoji, last_label, _ = render_result(dt_model, "🌴 ÁRBOL DE DECISIÓN", col_c)

    # Guardar en historial
    st.session_state.prediction_history.append({
        "emoji": last_emoji,
        "label": last_label,
        "time":  datetime.now().strftime("%H:%M:%S"),
    })
    if len(st.session_state.prediction_history) > 5:
        st.session_state.prediction_history.pop(0)

    # Tabla de características
    st.markdown("")
    with st.expander("📊 Ver características ingresadas"):
        feature_names = [
            "Pelo","Plumas","Huevos","Leche","Vuela","Acuático",
            "Depredador","Con dientes","Columna vertebral","Respira",
            "Venenoso","Aletas","Patas","Cola","Doméstico","Tamaño gato",
        ]
        df_feat = pd.DataFrame([{
            n: ("✅ Sí" if v == 1 else "❌ No") if n != "Patas" else int(v)
            for n, v in zip(feature_names, features[0])
        }])
        st.dataframe(df_feat, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  FOOTER — hierba animada
# ─────────────────────────────────────────────────────────────
heights_g = [32,50,38,56,28,46,42,54,34,50,40,58,30,44,38,52,44,36,50,34,48,42,56,38,48]
greens    = ["#2e7d32","#388e3c","#43a047","#1b5e20","#4caf50","#33691e","#558b2f","#27ae60"]
blades_html = "".join(
    f'<div class="blade" style="'
    f'width:{5+(i%3)*2}px;'
    f'height:{heights_g[i%len(heights_g)]}px;'
    f'background:linear-gradient(180deg,{greens[i%len(greens)]} 0%,{greens[(i+3)%len(greens)]} 100%);'
    f'animation-delay:{i*0.055:.2f}s;'
    f'animation-duration:{1.8+(i%5)*0.25:.1f}s;'
    f'opacity:{0.5+(i%6)*0.08:.2f};'
    f'"></div>'
    for i in range(140)
)
st.markdown(f'<div class="grass-row">{blades_html}</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:rgba(129,199,132,.3);'
    'font-size:.8rem;margin-top:.25rem;">'
    '🐾 Zoo Classifier &nbsp;·&nbsp; PA2 &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; 2025</p>',
    unsafe_allow_html=True,
)
