import streamlit as st
import joblib
import requests
import io
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
GITHUB_USER   = "juliocastrolimas16-boop"
GITHUB_REPO   = "zoo_PA2"
GITHUB_BRANCH = "main"
MODEL_FOLDER  = "model1"
COLAB_URL     = "https://colab.research.google.com/drive/1mzsUZ0WHfJH9Wf70WA4mVKY3dNTVa6K-?usp=sharing"

CLASS_LABELS = {
    "Mamifero":            ("🦁", "Mamífero",            "#D4A017"),
    "Mamífero":            ("🦁", "Mamífero",            "#D4A017"),
    "Ave":                 ("🦅", "Ave",                 "#4FC3F7"),
    "Pez":                 ("🐟", "Pez",                 "#26C6DA"),
    "Invertebrado_Marino": ("🦑", "Invertebrado Marino", "#AB47BC"),
    "Invertebrado marino": ("🦑", "Invertebrado Marino", "#AB47BC"),
    "Insecto":             ("🐛", "Insecto",             "#66BB6A"),
    "Reptil":              ("🦎", "Reptil",              "#FFA726"),
    "Anfibio":             ("🐸", "Anfibio",             "#26A69A"),
}

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
    for url in urls:
        try:
            resp = requests.get(url, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                content = resp.content
                if content[:10] == b"version ht" or b"oid sha256" in content[:200]:
                    continue
                return joblib.load(io.BytesIO(content))
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"No se pudo descargar {filename}.\nURLs probadas:\n"
        + "\n".join(f"  {u}" for u in urls)
        + f"\nUltimo error: {last_err}"
    )

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zoo Clasificador",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CSS — TEMA ZOOLOGICO SELVA
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Boogaloo&family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* Fondo selva oscura con patron de hojas */
[data-testid="stAppViewContainer"] {
    background-color: #071a07;
    background-image:
        radial-gradient(ellipse at 8%  15%, rgba(27,94,32,0.6)  0%, transparent 50%),
        radial-gradient(ellipse at 92% 85%, rgba(20,70,20,0.55) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(10,40,10,0.35) 0%, transparent 65%),
        radial-gradient(ellipse at 80% 10%, rgba(30,80,10,0.4)  0%, transparent 45%),
        radial-gradient(ellipse at 15% 90%, rgba(15,60,15,0.4)  0%, transparent 45%);
    min-height: 100vh;
}

/* Patron de patas en el fondo */
[data-testid="stAppViewContainer"]::before {
    content: "🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾  🐾";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    font-size: 3rem;
    line-height: 5rem;
    word-spacing: 2.5rem;
    opacity: 0.035;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
    padding: 1.5rem;
}

/* Hero banner */
.zoo-hero {
    position: relative;
    text-align: center;
    padding: 2.8rem 1rem 1rem;
    z-index: 1;
}
.zoo-hero-title {
    font-family: 'Boogaloo', cursive;
    font-size: clamp(2.8rem, 6.5vw, 5rem);
    color: #fff;
    text-shadow:
        0 0 40px rgba(255,210,60,0.5),
        4px 4px 0 #1b5e20,
        7px 7px 0 #0a3300;
    letter-spacing: .05em;
    margin: 0 0 .4rem;
    line-height: 1.1;
}
.zoo-hero-title .accent { color: #FFD54F; }
.zoo-hero-sub {
    color: #81c784;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 1.4rem;
    letter-spacing: .02em;
}

/* Valla madera */
.wood-rail {
    width: 100%;
    height: 8px;
    background: repeating-linear-gradient(
        90deg,
        #7B5314 0px, #7B5314 28px,
        #9B7A1A 28px, #9B7A1A 30px
    );
    border-radius: 4px;
    margin: .4rem 0;
    opacity: .65;
    box-shadow: 0 2px 8px rgba(0,0,0,.5);
}
.fence-row {
    display: flex;
    justify-content: space-between;
    padding: 0 2%;
    margin: 0;
}
.fence-post {
    width: 16px;
    background: linear-gradient(180deg, #9B7A1A 0%, #6B4F10 70%, #4a350a 100%);
    border-radius: 4px 4px 0 0;
    box-shadow: 2px 0 4px rgba(0,0,0,.4);
}

/* Cards de seccion */
.zoo-card {
    position: relative;
    background: linear-gradient(140deg,
        rgba(10,45,10,0.88) 0%,
        rgba(5,30,5,0.92) 100%);
    border: 1px solid rgba(100,200,100,0.15);
    border-radius: 20px;
    padding: 1.7rem 2rem 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow:
        0 6px 30px rgba(0,0,0,.5),
        inset 0 1px 0 rgba(165,214,167,.07),
        inset 0 -1px 0 rgba(0,0,0,.2);
}
.zoo-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 20px 20px 0 0;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(100,200,100,0.4) 30%,
        rgba(165,214,167,0.6) 50%,
        rgba(100,200,100,0.4) 70%,
        transparent 100%);
}
.zoo-card-title {
    font-family: 'Boogaloo', cursive;
    font-size: 1.3rem;
    color: #a5d6a7;
    letter-spacing: .06em;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.zoo-card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(100,200,100,.25), transparent);
    margin-left: .6rem;
}

/* Inputs */
label[data-testid="stWidgetLabel"] > div {
    color: #b9deba !important;
    font-weight: 700 !important;
    font-size: .87rem !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(0,50,0,0.6) !important;
    color: #e8f5e9 !important;
    border: 1px solid rgba(100,200,100,0.22) !important;
    border-radius: 10px !important;
}
[data-testid="stNumberInput"] input {
    background: rgba(0,50,0,0.6) !important;
    color: #e8f5e9 !important;
    border: 1px solid rgba(100,200,100,0.22) !important;
    border-radius: 10px !important;
}

/* Boton predecir */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #388e3c 0%, #1b5e20 50%, #2e7d32 100%);
    color: #f1f8e9;
    font-family: 'Boogaloo', cursive;
    font-size: 1.4rem;
    letter-spacing: .1em;
    border: 2px solid rgba(165,214,167,.3);
    border-radius: 18px;
    padding: 1rem 2rem;
    cursor: pointer;
    transition: all .25s ease;
    box-shadow: 0 6px 25px rgba(56,142,60,.4), inset 0 1px 0 rgba(255,255,255,.1);
    text-shadow: 0 2px 4px rgba(0,0,0,.4);
}
div.stButton > button:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 14px 40px rgba(56,142,60,.6);
    border-color: rgba(165,214,167,.55);
    background: linear-gradient(135deg, #43a047 0%, #2e7d32 50%, #388e3c 100%);
}

/* Resultado */
.result-card {
    border-radius: 22px;
    padding: 2.2rem 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,.5);
    transition: transform .2s;
}
.result-card:hover { transform: translateY(-4px); }
.result-card::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,.06);
    pointer-events: none;
}
.result-card::after {
    content: '';
    position: absolute;
    bottom: -35px; left: -35px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: rgba(255,255,255,.04);
    pointer-events: none;
}
.result-model { font-size:.78rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; opacity:.7; margin-bottom:.7rem; }
.result-emoji { font-size:4.2rem; line-height:1; margin-bottom:.5rem; filter: drop-shadow(0 6px 12px rgba(0,0,0,.6)); }
.result-label { font-family:'Boogaloo',cursive; font-size:2.3rem; color:#fff; text-shadow:0 3px 10px rgba(0,0,0,.6); }
.result-conf  { font-size:.88rem; opacity:.8; margin-top:.5rem; }
.result-bar-wrap { margin:.7rem auto 0; width:75%; height:7px; background:rgba(255,255,255,.15); border-radius:4px; }
.result-bar { height:100%; background:rgba(255,255,255,.75); border-radius:4px; transition:width .7s cubic-bezier(.4,0,.2,1); }

.rf-card { background: linear-gradient(140deg, #1a5c1e 0%, #2e7d32 60%, #1b5e20 100%); border:1px solid rgba(165,214,167,.25); color:#e8f5e9; }
.dt-card { background: linear-gradient(140deg, #33691e 0%, #558b2f 60%, #3d7a22 100%); border:1px solid rgba(220,237,200,.25); color:#f1f8e9; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071a07 0%, #0a2e0a 50%, #071a07 100%) !important;
    border-right: 1px solid rgba(100,200,100,.1) !important;
}
[data-testid="stSidebar"] * { color: #b9deba !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #81c784 !important; }

.clase-chip {
    display: flex;
    align-items: center;
    gap: .45rem;
    background: rgba(50,130,50,0.1);
    border: 1px solid rgba(100,200,100,.12);
    border-left: 3px solid rgba(100,200,100,.4);
    border-radius: 0 8px 8px 0;
    padding: .32rem .75rem;
    margin-bottom: .28rem;
    font-size: .88rem;
    color: #b9deba;
    transition: background .2s;
}
.clase-chip:hover { background: rgba(50,130,50,0.2); }

/* Colab badge */
.colab-link {
    display: flex;
    align-items: center;
    gap: .5rem;
    justify-content: center;
    background: linear-gradient(135deg, #F9AB00 0%, #E37400 100%);
    color: #fff !important;
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    font-size: .92rem;
    padding: .6rem 1.2rem;
    border-radius: 12px;
    text-decoration: none !important;
    box-shadow: 0 4px 20px rgba(249,171,0,.4);
    transition: all .2s;
    margin-bottom: .4rem;
}
.colab-link:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(249,171,0,.6);
}

/* Hierba */
.grass-row {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 3px;
    margin-top: 2rem;
    height: 60px;
    overflow: hidden;
}
.blade {
    border-radius: 4px 4px 0 0;
    animation: sway 2.2s ease-in-out infinite alternate;
}
@keyframes sway {
    0%   { transform: rotate(-10deg); transform-origin: bottom center; }
    100% { transform: rotate( 10deg); transform-origin: bottom center; }
}

/* Divisor madera */
.wood-divider {
    width:100%; height:5px;
    background: repeating-linear-gradient(90deg,
        #5a3e0c 0px, #5a3e0c 25px,
        #7B5314 25px, #7B5314 27px);
    border-radius:3px; margin:1.6rem 0; opacity:.5;
}

[data-testid="stAlert"] { border-radius:14px !important; }
[data-testid="stExpander"] {
    background: rgba(10,40,10,0.6) !important;
    border: 1px solid rgba(100,200,100,.12) !important;
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Panel de control")
    st.markdown("---")

    model_choice = st.radio(
        "🤖 Modelo a usar",
        ["🌲 Ambos modelos", "🌳 Random Forest", "🌴 Árbol de Decisión"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🦒 Clases del zoo")
    clases = [
        ("🦁","Mamífero"),("🦅","Ave"),("🐟","Pez"),
        ("🦑","Invertebrado Marino"),("🐛","Insecto"),
        ("🦎","Reptil"),("🐸","Anfibio"),
    ]
    for emoji, nombre in clases:
        st.markdown(f'<div class="clase-chip">{emoji} {nombre}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📓 Notebook de entrenamiento")
    st.markdown(
        f'<a class="colab-link" href="{COLAB_URL}" target="_blank">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" height="18"/>'
        f'&nbsp;Abrir en Colab</a>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Aquí puedes ver el entrenamiento completo de los modelos "
        "Random Forest y Árbol de Decisión sobre el dataset Zoo."
    )
    st.markdown("")
    st.markdown("### 📊 Dataset")
    st.caption("161 animales · 7 clases · 16 características")


# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="zoo-hero">
  <div class="zoo-hero-title">🦁 Zoo <span class="accent">Clasificador</span> 🐾</div>
  <div class="zoo-hero-sub">Identifica la especie de cualquier animal usando Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# Boton Colab centrado bajo el titulo
col_l, col_m, col_r = st.columns([1.5, 1, 1.5])
with col_m:
    st.link_button("📓 Ver Notebook Colab", COLAB_URL, use_container_width=True)

# Valla decorativa
posts = 55
post_heights = [38,44,40,48,36,44,42,50,38,46,40,52,36,44,42] * 4
posts_html = "".join(
    f'<div class="fence-post" style="height:{post_heights[i%len(post_heights)]}px;'
    f'margin-top:{max(0,52-post_heights[i%len(post_heights)])}px"></div>'
    for i in range(posts)
)
st.markdown(f"""
<div style="padding: .5rem 1rem 0; overflow:hidden;">
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
#  FORMULARIO DE ENTRADA
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="wood-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<h3 style="font-family:Boogaloo,cursive;color:#81c784;letter-spacing:.06em;margin-bottom:.2rem;">'
    '🔬 Características del animal</h3>',
    unsafe_allow_html=True,
)

BINARY_OPTIONS = {0: "❌  No  (0)", 1: "✅  Sí  (1)"}

def binary_select(label, key, col):
    with col:
        return st.selectbox(label, options=[0, 1],
                            format_func=lambda x: BINARY_OPTIONS[x], key=key)

# ── Bloque 1: Cobertura & reproducción
st.markdown('<div class="zoo-card"><div class="zoo-card-title">🧬 Cobertura corporal & reproducción</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
pelo   = binary_select("🐾 Pelo",   "pelo",   c1)
plumas = binary_select("🪶 Plumas", "plumas", c2)
huevos = binary_select("🥚 Huevos", "huevos", c3)
leche  = binary_select("🍼 Leche",  "leche",  c4)
st.markdown('</div>', unsafe_allow_html=True)

# ── Bloque 2: Capacidades & hábitat
st.markdown('<div class="zoo-card"><div class="zoo-card-title">🌍 Capacidades & hábitat</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
vuela       = binary_select("🦋 Vuela",       "vuela",      c1)
acuatico    = binary_select("🌊 Acuático",    "acuatico",   c2)
depredador  = binary_select("🦷 Depredador",  "depredador", c3)
con_dientes = binary_select("😬 Con dientes", "con_dientes",c4)
st.markdown('</div>', unsafe_allow_html=True)

# ── Bloque 3: Fisiología
st.markdown('<div class="zoo-card"><div class="zoo-card-title">🔬 Fisiología & morfología</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
columna_vertebral = binary_select("🦴 Columna vert.",    "columna_vertebral", c1)
respira           = binary_select("💨 Respira (pulmón)", "respira",           c2)
venenoso          = binary_select("☠️ Venenoso",         "venenoso",          c3)
aletas            = binary_select("🐠 Aletas",           "aletas",            c4)
cola              = binary_select("🐉 Cola",             "cola",              c5)
st.markdown('</div>', unsafe_allow_html=True)

# ── Bloque 4: Otras
st.markdown('<div class="zoo-card"><div class="zoo-card-title">🏠 Otras características</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    patas = st.number_input("🦵 Número de patas", min_value=0, max_value=10,
                            value=4, step=1, key="patas")
domestico   = binary_select("🏡 Doméstico",      "domestico",   c2)
tamano_gato = binary_select("📏 Tamaño de gato", "tamano_gato", c3)
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  BOTON PREDECIR
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="wood-divider"></div>', unsafe_allow_html=True)
predict_btn = st.button("🔍  ¡Identificar animal!", use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  RESULTADO
# ─────────────────────────────────────────────────────────────
if predict_btn:
    features = np.array([[
        pelo, plumas, huevos, leche, vuela, acuatico,
        depredador, con_dientes, columna_vertebral, respira,
        venenoso, aletas, patas, cola, domestico, tamano_gato,
    ]])

    st.markdown(
        '<h3 style="font-family:Boogaloo,cursive;color:#81c784;letter-spacing:.06em;margin-top:1.2rem;">'
        '🎯 Resultado del análisis</h3>',
        unsafe_allow_html=True,
    )

    res_col1, _, res_col2 = st.columns([1, .06, 1])

    def show_result(model, model_name, card_class, col):
        pred     = model.predict(features)[0]
        pred_key = str(pred).strip()
        info     = CLASS_LABELS.get(pred_key, ("🐾", pred_key, "#4caf50"))
        emoji, label, _ = info

        conf_html = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            conf  = proba.max() * 100
            conf_html = f"""
            <div class="result-conf">Confianza: <b>{conf:.1f}%</b></div>
            <div class="result-bar-wrap">
              <div class="result-bar" style="width:{int(conf)}%"></div>
            </div>"""

        with col:
            st.markdown(f"""
            <div class="result-card {card_class}">
              <div class="result-model">{model_name}</div>
              <div class="result-emoji">{emoji}</div>
              <div class="result-label">{label}</div>
              {conf_html}
            </div>
            """, unsafe_allow_html=True)

    show_result(rf_model, "🌲 Random Forest",     "rf-card", res_col1)
    show_result(dt_model, "🌴 Árbol de Decisión", "dt-card", res_col2)

    st.markdown("")
    with st.expander("📊 Ver características ingresadas"):
        feature_names = [
            "pelo","plumas","huevos","leche","vuela","acuatico",
            "depredador","con_dientes","columna_vertebral","respira",
            "venenoso","aletas","patas","cola","domestico","tamano_gato",
        ]
        feat_data = {n: [int(v)] for n, v in zip(feature_names, features[0])}
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  FOOTER — hierba animada
# ─────────────────────────────────────────────────────────────
heights_g  = [32,48,38,55,28,46,42,52,35,50,40,58,30,45,38,52,44,36,50,34,48,42,56,38,46]
greens     = ["#2e7d32","#388e3c","#43a047","#1b5e20","#4caf50","#33691e","#558b2f"]
blades_html = "".join(
    f'<div class="blade" style="'
    f'width:{5 + (i%3)*2}px;'
    f'height:{heights_g[i % len(heights_g)]}px;'
    f'background:linear-gradient(180deg,{greens[i%len(greens)]} 0%,{greens[(i+2)%len(greens)]} 100%);'
    f'animation-delay:{i * 0.06:.2f}s;'
    f'animation-duration:{1.8 + (i%4)*0.3:.1f}s;'
    f'opacity:{0.55 + (i%5)*0.09:.2f};'
    f'"></div>'
    for i in range(130)
)
st.markdown(f'<div class="grass-row">{blades_html}</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:rgba(129,199,132,.35);'
    'font-size:.82rem;margin-top:.2rem;font-family:Nunito,sans-serif;">'
    '🐾 Zoo Clasificador &nbsp;·&nbsp; PA2 &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; 2025</p>',
    unsafe_allow_html=True,
)
