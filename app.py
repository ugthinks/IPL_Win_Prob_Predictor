from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "models" / "logistic_model.pkl"

FEATURES = [
    "batting_team",
    "bowling_team",
    "city",
    "runs_left",
    "balls_left",
    "wickets_left",
    "total_runs_x",
    "crr",
    "rrr",
]

TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

CITIES = [
    "Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Hyderabad", "Jaipur",
    "Kolkata", "Lucknow", "Mumbai", "Pune", "Mohali", "Chandigarh",
    "Dharamsala", "Visakhapatnam", "Dubai", "Abu Dhabi", "Sharjah",
]

CITY_ALIASES = {"Bangalore": "Bengaluru"}

TEAM_COLORS = {
    "Chennai Super Kings": "#FFCB05",
    "Delhi Capitals": "#2D8CFF",
    "Gujarat Titans": "#6FA8DC",
    "Kolkata Knight Riders": "#8B5CF6",
    "Lucknow Super Giants": "#27C4F4",
    "Mumbai Indians": "#2F7BFF",
    "Punjab Kings": "#FF3B4E",
    "Rajasthan Royals": "#FF5FB7",
    "Royal Challengers Bengaluru": "#FF2D2D",
    "Sunrisers Hyderabad": "#FF8A1F",
}

TOTAL_BALLS = 120


# --------------------------------------------------------------------------
# Model + cricket logic
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def overs_to_balls(overs):
    """Cricket notation: 12.3 means 12 overs and 3 balls = 75 legal balls."""
    overs = float(overs)
    completed_overs = int(overs)
    balls = int(round((overs - completed_overs) * 10))
    if not 0 <= balls <= 5:
        raise ValueError("The ball part of the overs must be between 0 and 5.")
    return completed_overs * 6 + balls


def compute_features(target_score, current_score, wickets_lost, balls_bowled):
    runs_left = target_score - current_score
    balls_left = TOTAL_BALLS - balls_bowled
    wickets_left = 10 - wickets_lost
    crr = (current_score / (balls_bowled / 6)) if balls_bowled > 0 else 0.0
    rrr = (runs_left / balls_left) * 6 if balls_left > 0 else 0.0
    return {
        "runs_left": int(runs_left),
        "balls_left": int(balls_left),
        "wickets_left": int(wickets_left),
        "total_runs_x": int(target_score),
        "crr": float(crr),
        "rrr": float(rrr),
    }


def predict(model, batting_team, bowling_team, city, feats):
    input_df = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": CITY_ALIASES.get(city, city),
        "runs_left": feats["runs_left"],
        "balls_left": feats["balls_left"],
        "wickets_left": feats["wickets_left"],
        "total_runs_x": feats["total_runs_x"],
        "crr": feats["crr"],
        "rrr": feats["rrr"],
    }])[FEATURES]

    if not np.isfinite(input_df.select_dtypes("number").to_numpy(dtype=float)).all():
        raise ValueError("Inputs produced an invalid number.")

    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    chasing = float(probabilities[classes.index(1)])
    bowling = float(probabilities[classes.index(0)])
    return chasing * 100, bowling * 100


def overs_label(balls):
    return f"{balls // 6}.{balls % 6}"


# --------------------------------------------------------------------------
# Visual layer (kept separate from the prediction logic above)
# --------------------------------------------------------------------------
CURSOR_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='34' height='34' viewBox='0 0 34 34'>"
    "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
    "<stop offset='0' stop-color='%23FF3D81'/><stop offset='1' stop-color='%23FFC23C'/>"
    "</linearGradient></defs>"
    "<path d='M4 3 L4 27 L10.5 21 L15 31 L19.5 29 L15 19.5 L24 19.5 Z' "
    "fill='url(%23g)' stroke='white' stroke-width='1.8' stroke-linejoin='round'/>"
    "<circle cx='26' cy='8' r='4' fill='%23E4002B' stroke='white' stroke-width='1.4'/>"
    "</svg>"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Manrope:wght@400;500;600;700&display=swap');

@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }

:root {
  --ink: #F4F1FF;
  --muted: #A9A4CF;
  --night: #0A0820;
  --glass: rgba(255,255,255,0.055);
  --edge: rgba(255,255,255,0.14);
  --ball: #FF3D81;
  --gold: #FFC23C;
}

html, body, .stApp, .stApp * { cursor: url("__CURSOR__") 4 3, auto !important; }
.stApp { font-family: 'Manrope', sans-serif; color: var(--ink); }

.stApp {
  background:
    radial-gradient(900px 600px at 12% -10%, rgba(255,61,129,0.30), transparent 60%),
    radial-gradient(800px 600px at 95% 5%, rgba(66,99,255,0.34), transparent 60%),
    radial-gradient(900px 700px at 50% 120%, rgba(255,194,60,0.20), transparent 60%),
    var(--night);
  background-attachment: fixed;
}
.stApp::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    radial-gradient(1.5px 1.5px at 20% 30%, rgba(255,255,255,.7), transparent 50%),
    radial-gradient(1.5px 1.5px at 70% 20%, rgba(255,255,255,.5), transparent 50%),
    radial-gradient(1px 1px at 40% 80%, rgba(255,255,255,.6), transparent 50%),
    radial-gradient(1px 1px at 85% 65%, rgba(255,255,255,.5), transparent 50%),
    radial-gradient(1.5px 1.5px at 10% 70%, rgba(255,255,255,.4), transparent 50%);
  animation: drift 40s linear infinite;
}
@keyframes drift { to { transform: translateY(-60px); } }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; position: relative; z-index: 1; }

/* Hero */
.hero { text-align: left; margin-bottom: 1.6rem; animation: rise .9s cubic-bezier(.2,.8,.2,1) both; }
@keyframes rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
.hero h1 {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: clamp(2.4rem, 6vw, 4.6rem); line-height: .98; letter-spacing: -0.03em; margin: 0;
  background: linear-gradient(100deg, #fff 10%, #FFC23C 45%, #FF3D81 80%);
  background-size: 200% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: sheen 7s ease-in-out infinite alternate;
}
@keyframes sheen { to { background-position: 100% 0; } }
.hero p { color: var(--muted); font-size: 1.05rem; max-width: 560px; margin: .9rem 0 0; line-height: 1.55; }
.live-dot { display:inline-flex; align-items:center; gap:.5rem; font-weight:600; font-size:.9rem; color: var(--ink);
  background: var(--glass); border:1px solid var(--edge); padding:.35rem .8rem; border-radius:99px; margin-bottom:1rem; }
.live-dot i { width:9px; height:9px; border-radius:50%; background: var(--ball); box-shadow:0 0 0 0 rgba(255,61,129,.8); animation: ping 1.6s infinite; }
@keyframes ping { 70% { box-shadow: 0 0 0 11px rgba(255,61,129,0); } 100% { box-shadow: 0 0 0 0 rgba(255,61,129,0); } }

/* Panels (Streamlit bordered containers) */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--glass); border: 1px solid var(--edge) !important; border-radius: 26px;
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  box-shadow: 0 30px 80px -30px rgba(0,0,0,.7), inset 0 1px 0 rgba(255,255,255,.12);
  transition: transform .35s ease, border-color .35s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: rgba(255,194,60,.45) !important; }

.panel-title { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.25rem; margin:.1rem 0 .4rem; }

/* Widgets */
label, .stSelectbox label p, .stSlider label p, .stNumberInput label p, .stSelectSlider label p {
  color: var(--muted) !important; font-weight: 600 !important; font-size: .88rem !important;
}
div[data-baseweb="select"] > div, .stNumberInput input, div[data-baseweb="input"] > div {
  background: rgba(255,255,255,.07) !important; border: 1px solid var(--edge) !important;
  border-radius: 14px !important; color: var(--ink) !important; transition: all .25s ease;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {
  border-color: var(--gold) !important; box-shadow: 0 0 0 4px rgba(255,194,60,.14); transform: translateY(-1px);
}
div[data-baseweb="select"] * , .stNumberInput input { color: var(--ink) !important; }
ul[role="listbox"], div[data-baseweb="popover"] > div { background: #17123F !important; border-radius: 14px !important; }
li[role="option"]:hover { background: rgba(255,61,129,.25) !important; }

div[data-testid="stSlider"] [role="slider"] {
  background: linear-gradient(135deg, var(--ball), var(--gold)) !important; border: 3px solid #fff !important;
  box-shadow: 0 0 0 6px rgba(255,61,129,.25), 0 6px 18px rgba(255,61,129,.6); transition: transform .2s ease;
}
div[data-testid="stSlider"] [role="slider"]:hover { transform: scale(1.25); }
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: rgba(255,255,255,.12); }
div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="linear-gradient"] { background: linear-gradient(90deg, var(--ball), var(--gold)) !important; }
div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"], div[data-testid="stSliderThumbValue"] { color: var(--gold) !important; font-weight: 700; }

/* Match banner */
.versus { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:1.4rem; }
.side { flex:1; }
.side .role { font-size:.8rem; color: var(--muted); font-weight:600; }
.side .name { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:clamp(1.1rem,2.4vw,1.7rem); line-height:1.1; margin-top:.2rem; }
.side.right { text-align:right; }
.vs { width:48px; height:48px; border-radius:50%; display:grid; place-items:center; font-weight:800;
  background: radial-gradient(circle at 30% 30%, #FF6B9C, #C1124F); box-shadow: 0 0 30px rgba(255,61,129,.6); animation: spin 14s linear infinite; }
.vs span { animation: spin 14s linear infinite reverse; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Probability ring */
.hero-stat { display:flex; align-items:center; gap:2rem; flex-wrap:wrap; justify-content:center; }
.ring { --c: #FFC23C; width: 230px; height: 230px; border-radius: 50%; display:grid; place-items:center; position:relative;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.09) 0);
  animation: fill 1.4s cubic-bezier(.2,.8,.2,1) forwards; filter: drop-shadow(0 0 28px var(--c)); }
@keyframes fill { from { --p: 0; } }
.ring::after { content:""; position:absolute; inset:16px; border-radius:50%; background: #120E34; box-shadow: inset 0 0 40px rgba(0,0,0,.6); }
.ring .inner { position:relative; z-index:1; text-align:center; }
.ring .num { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:3.4rem; line-height:1; }
.ring .num small { font-size:1.5rem; opacity:.7; }
.ring .cap { font-size:.82rem; color: var(--muted); margin-top:.35rem; font-weight:600; }

.verdict { flex:1; min-width:240px; }
.verdict h3 { font-family:'Bricolage Grotesque',sans-serif; font-size:1.9rem; font-weight:800; margin:0 0 .4rem; line-height:1.1; }
.verdict p { color: var(--muted); margin:0; line-height:1.55; }

/* Tug of war */
.tug { margin-top:1.8rem; }
.tug-head { display:flex; justify-content:space-between; font-weight:700; margin-bottom:.55rem; font-size:.95rem; }
.tug-bar { height: 18px; border-radius: 99px; background: rgba(255,255,255,.08); overflow:hidden; display:flex; position:relative; }
.tug-bar .a { height:100%; transition: width 1s cubic-bezier(.2,.8,.2,1); animation: grow 1.2s cubic-bezier(.2,.8,.2,1); position:relative; }
.tug-bar .a::after { content:""; position:absolute; right:-1px; top:0; bottom:0; width:4px; background:#fff; box-shadow:0 0 14px 3px #fff; }
.tug-bar .b { height:100%; flex:1; }
@keyframes grow { from { width: 0 !important; } }

/* Stat chips */
.chips { display:grid; grid-template-columns: repeat(4, 1fr); gap:.9rem; margin-top:1.6rem; }
.chip { padding:1rem 1.1rem; border-radius:18px; background: rgba(255,255,255,.05); border:1px solid var(--edge);
  transition: transform .25s ease, background .25s ease, border-color .25s ease; }
.chip:hover { transform: translateY(-5px) rotate(-.6deg); background: rgba(255,255,255,.1); border-color: var(--gold); }
.chip .v { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.7rem; }
.chip .l { color: var(--muted); font-size:.82rem; font-weight:600; margin-top:.15rem; }
@media (max-width: 760px) { .chips { grid-template-columns: repeat(2, 1fr); } .ring { width:190px; height:190px; } .ring .num { font-size:2.7rem; } }

/* Notices */
.notice { padding:1.1rem 1.3rem; border-radius:18px; border:1px solid var(--edge); background: rgba(255,255,255,.06); line-height:1.5; }
.notice.warn { border-color: rgba(255,194,60,.55); background: rgba(255,194,60,.08); }
.notice.win  { border-color: rgba(60,255,160,.55); background: rgba(60,255,160,.08); }
.notice b { font-family:'Bricolage Grotesque',sans-serif; font-size:1.15rem; }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""".replace("__CURSOR__", CURSOR_SVG)

CURSOR_FX = """
<script>
(function () {
  try {
    const doc = window.parent.document;
    if (doc.getElementById("cb-orb")) return;
    const st = doc.createElement("style");
    st.textContent = `
      #cb-orb, #cb-tail { position: fixed; top: 0; left: 0; pointer-events: none; z-index: 99999; border-radius: 50%; }
      #cb-orb { width: 38px; height: 38px; margin: -19px 0 0 -19px; border: 2px solid rgba(255,194,60,.9);
        box-shadow: 0 0 22px rgba(255,194,60,.7), inset 0 0 14px rgba(255,61,129,.55); transition: width .2s, height .2s, margin .2s, border-color .2s; mix-blend-mode: screen; }
      #cb-orb.hot { width: 64px; height: 64px; margin: -32px 0 0 -32px; border-color: rgba(255,61,129,.95); }
      #cb-tail { width: 120px; height: 120px; margin: -60px 0 0 -60px;
        background: radial-gradient(circle, rgba(255,61,129,.22), transparent 65%); }
      .cb-burst { position: fixed; pointer-events: none; z-index: 99998; width: 10px; height: 10px; margin: -5px 0 0 -5px;
        border-radius: 50%; border: 2px solid #FFC23C; animation: cbb .7s ease-out forwards; }
      @keyframes cbb { to { transform: scale(7); opacity: 0; } }
    `;
    doc.head.appendChild(st);
    const orb = doc.createElement("div"); orb.id = "cb-orb";
    const tail = doc.createElement("div"); tail.id = "cb-tail";
    doc.body.appendChild(tail); doc.body.appendChild(orb);
    let x = -100, y = -100, ox = x, oy = y, tx = x, ty = y;
    doc.addEventListener("mousemove", e => { x = e.clientX; y = e.clientY; });
    doc.addEventListener("mouseover", e => {
      const hot = e.target.closest && e.target.closest('[data-baseweb="select"], [role="slider"], input, button, .chip');
      orb.classList.toggle("hot", !!hot);
    });
    doc.addEventListener("mousedown", e => {
      const b = doc.createElement("div"); b.className = "cb-burst";
      b.style.left = e.clientX + "px"; b.style.top = e.clientY + "px";
      doc.body.appendChild(b); setTimeout(() => b.remove(), 750);
    });
    (function loop() {
      ox += (x - ox) * 0.22; oy += (y - oy) * 0.22;
      tx += (x - tx) * 0.07; ty += (y - ty) * 0.07;
      orb.style.transform = `translate(${ox}px, ${oy}px)`;
      tail.style.transform = `translate(${tx}px, ${ty}px)`;
      requestAnimationFrame(loop);
    })();
  } catch (err) { /* cursor effects are optional */ }
})();
</script>
"""


def render_result(batting_team, bowling_team, chasing_pct, bowling_pct, feats, balls_bowled, current_score, wickets_lost):
    color = TEAM_COLORS.get(batting_team, "#FFC23C")
    bowl_color = TEAM_COLORS.get(bowling_team, "#FF3D81")

    if chasing_pct >= 80:
        headline, note = f"{batting_team} are cruising", "The chase is firmly under control."
    elif chasing_pct >= 60:
        headline, note = f"{batting_team} have the edge", "The chase is ahead, but a couple of wickets would flip it."
    elif chasing_pct >= 40:
        headline, note = "Anyone's game", "This chase is balanced on a knife-edge."
    elif chasing_pct >= 20:
        headline, note = f"{bowling_team} are on top", "The chasing side needs a big over."
    else:
        headline, note = f"{bowling_team} are in control", "The chasing side needs something special."

    st.markdown(
        f"""
        <div class="versus">
          <div class="side"><div class="role">Chasing</div><div class="name" style="color:{color}">{batting_team}</div></div>
          <div class="vs"><span>VS</span></div>
          <div class="side right"><div class="role">Defending</div><div class="name" style="color:{bowl_color}">{bowling_team}</div></div>
        </div>

        <div class="hero-stat">
          <div class="ring" style="--p:{chasing_pct:.2f}; --c:{color}">
            <div class="inner">
              <div class="num">{chasing_pct:.1f}<small>%</small></div>
              <div class="cap">{batting_team} win</div>
            </div>
          </div>
          <div class="verdict">
            <h3>{headline}</h3>
            <p>{note} {current_score}/{wickets_lost} after {overs_label(balls_bowled)} overs.</p>
          </div>
        </div>

        <div class="tug">
          <div class="tug-head">
            <span style="color:{color}">{batting_team} {chasing_pct:.1f}%</span>
            <span style="color:{bowl_color}">{bowling_pct:.1f}% {bowling_team}</span>
          </div>
          <div class="tug-bar">
            <div class="a" style="width:{chasing_pct:.2f}%; background:linear-gradient(90deg,{color}88,{color})"></div>
            <div class="b" style="background:linear-gradient(90deg,{bowl_color}, {bowl_color}88)"></div>
          </div>
        </div>

        <div class="chips">
          <div class="chip"><div class="v">{feats['runs_left']}</div><div class="l">Runs needed</div></div>
          <div class="chip"><div class="v">{feats['balls_left']}</div><div class="l">Balls left</div></div>
          <div class="chip"><div class="v">{feats['crr']:.2f}</div><div class="l">Current run rate</div></div>
          <div class="chip"><div class="v">{feats['rrr']:.2f}</div><div class="l">Required run rate</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def notice(kind, title, body):
    st.markdown(
        f'<div class="notice {kind}"><b>{title}</b><br>{body}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
st.set_page_config(page_title="IPL Chase Predictor", page_icon="🏏", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
components.html(CURSOR_FX, height=0)

st.markdown(
    """
    <div class="hero">
      <div class="live-dot"><i></i>Second innings, live</div>
      <h1>Will they chase it down?</h1>
      <p>Set the match situation and see each side's chance of winning, updated the moment you change anything.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Could not find `models/logistic_model.pkl`. Make sure it is in the repository next to `app.py`.")
    st.stop()
except Exception as exc:
    st.error(f"The model file could not be loaded: {exc}")
    st.stop()

left, right = st.columns([5, 7], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Match setup</div>', unsafe_allow_html=True)
        batting_team = st.selectbox("Chasing team", TEAMS, index=TEAMS.index("Mumbai Indians"))
        bowling_options = [t for t in TEAMS if t != batting_team]
        bowling_team = st.selectbox("Bowling team", bowling_options, index=bowling_options.index("Chennai Super Kings") if "Chennai Super Kings" in bowling_options else 0)
        city = st.selectbox("Venue city", CITIES, index=CITIES.index("Mumbai"))

        st.markdown('<div class="panel-title" style="margin-top:1rem">Scoreboard</div>', unsafe_allow_html=True)
        target_score = st.number_input("Target score", min_value=1, max_value=300, value=185, step=1)
        current_score = st.slider("Current score", 0, int(target_score), min(95, int(target_score)))
        wickets_lost = st.select_slider("Wickets lost", options=list(range(0, 11)), value=3)
        c1, c2 = st.columns(2)
        with c1:
            overs_done = st.select_slider("Overs completed", options=list(range(0, 21)), value=11)
        with c2:
            extra_balls = st.select_slider("Balls in current over", options=[0, 1, 2, 3, 4, 5], value=2)

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Win probability</div>', unsafe_allow_html=True)

        balls_bowled = overs_to_balls(float(f"{overs_done}.{extra_balls}"))
        target_score = int(target_score)
        current_score = int(current_score)

        if batting_team == bowling_team:
            notice("warn", "Pick two different teams", "The chasing and bowling teams can't be the same.")
        elif balls_bowled > TOTAL_BALLS:
            notice("warn", "Too many overs", "An innings is 20 overs. Reduce the overs or balls.")
        elif current_score >= target_score:
            notice("win", f"{batting_team} have reached the target", f"{current_score} against a target of {target_score}. The chase is complete.")
        elif wickets_lost >= 10:
            notice("warn", f"{batting_team} are all out", f"{bowling_team} win. The innings ended on {current_score}, short of {target_score}.")
        elif balls_bowled >= TOTAL_BALLS:
            notice("warn", f"{bowling_team} win", f"The 20 overs are done and {batting_team} finished {target_score - current_score} runs short.")
        else:
            try:
                feats = compute_features(target_score, current_score, wickets_lost, balls_bowled)
                chasing_pct, bowling_pct = predict(model, batting_team, bowling_team, city, feats)
                render_result(batting_team, bowling_team, chasing_pct, bowling_pct, feats, balls_bowled, current_score, wickets_lost)
            except Exception as exc:
                notice("warn", "Couldn't calculate a prediction", f"Please check the match inputs. ({exc})")