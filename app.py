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

CHASE_COLOR = "#4F7DF3"
DEFEND_COLOR = "#E3B341"

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
    "<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 32 32'>"
    "<path d='M4 3 L4 26 L10 20.5 L14 29.5 L18 27.8 L14 19 L22.5 19 Z' "
    "fill='%23E8497F' stroke='white' stroke-width='1.6' stroke-linejoin='round'/>"
    "<circle cx='25' cy='8' r='3.6' fill='%23C8283F' stroke='white' stroke-width='1.2'/>"
    "</svg>"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Manrope:wght@400;500;600;700&display=swap');

@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }

:root {
  --ink: #EEF0F8;
  --muted: #9AA0BC;
  --bg: #0E1120;
  --panel: #151932;
  --edge: rgba(255,255,255,0.09);
  --chase: #4F7DF3;
  --defend: #E3B341;
  --accent: #E8497F;
}

html, body, .stApp, .stApp * { cursor: url("__CURSOR__") 4 3, auto !important; }
.stApp { font-family: 'Manrope', sans-serif; color: var(--ink); }

.stApp {
  background: radial-gradient(1000px 500px at 50% -15%, rgba(79,125,243,0.07), transparent 70%), var(--bg);
  background-attachment: fixed;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; position: relative; z-index: 1; }

/* Hero */
.hero { text-align: left; margin-bottom: 1.6rem; animation: rise .7s ease-out both; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.hero h1 {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: clamp(2.3rem, 5.6vw, 4.3rem); line-height: 1; letter-spacing: -0.025em; margin: 0; color: var(--ink);
}
.hero p { color: var(--muted); font-size: 1.02rem; max-width: 560px; margin: .9rem 0 0; line-height: 1.55; }
.live-dot { display:inline-flex; align-items:center; gap:.5rem; font-weight:600; font-size:.85rem; color: var(--muted);
  background: rgba(255,255,255,.04); border:1px solid var(--edge); padding:.3rem .7rem; border-radius:8px; margin-bottom:1rem; }
.live-dot i { width:8px; height:8px; border-radius:50%; background: var(--accent); animation: pulse 2s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: .35; } }

/* Panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel); border: 1px solid var(--edge) !important; border-radius: 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,.28);
}

.panel-title { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.2rem; margin:.1rem 0 .4rem; }

/* Widgets */
label, .stSelectbox label p, .stSlider label p, .stNumberInput label p, .stSelectSlider label p {
  color: var(--muted) !important; font-weight: 600 !important; font-size: .86rem !important;
}
div[data-baseweb="select"] > div, .stNumberInput input, div[data-baseweb="input"] > div {
  background: rgba(255,255,255,.04) !important; border: 1px solid var(--edge) !important;
  border-radius: 8px !important; color: var(--ink) !important; transition: border-color .2s ease;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover { border-color: rgba(255,255,255,.28) !important; }
div[data-baseweb="select"] * , .stNumberInput input { color: var(--ink) !important; }
ul[role="listbox"], div[data-baseweb="popover"] > div { background: #1B2040 !important; border-radius: 8px !important; }
li[role="option"]:hover { background: rgba(79,125,243,.22) !important; }

div[data-testid="stSlider"] [role="slider"] {
  background: var(--accent) !important; border: 2px solid #fff !important; box-shadow: 0 2px 6px rgba(0,0,0,.4);
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: rgba(255,255,255,.12); }
div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="linear-gradient"] { background: var(--accent) !important; }
div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"], div[data-testid="stSliderThumbValue"] { color: var(--muted) !important; font-weight: 600; }

/* Match banner */
.versus { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:1.4rem; }
.side { flex:1; }
.side .role { font-size:.8rem; color: var(--muted); font-weight:600; }
.side .name { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:clamp(1.1rem,2.3vw,1.65rem); line-height:1.1; margin-top:.2rem; }
.side.right { text-align:right; }
.vs { width:40px; height:40px; border-radius:50%; display:grid; place-items:center; font-weight:700; font-size:.8rem;
  background: var(--accent); color:#fff; }

/* Probability ring */
.hero-stat { display:flex; align-items:center; gap:2rem; flex-wrap:wrap; justify-content:center; }
.ring { --c: var(--chase); width: 230px; height: 230px; border-radius: 50%; display:grid; place-items:center; position:relative;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.08) 0);
  animation: fill 1.1s cubic-bezier(.2,.8,.2,1) forwards; }
@keyframes fill { from { --p: 0; } }
.ring::after { content:""; position:absolute; inset:16px; border-radius:50%; background: var(--panel); }
.ring .inner { position:relative; z-index:1; text-align:center; }
.ring .num { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:3.4rem; line-height:1; }
.ring .num small { font-size:1.5rem; opacity:.65; }
.ring .cap { font-size:.82rem; color: var(--muted); margin-top:.35rem; font-weight:600; }

.verdict { flex:1; min-width:240px; }
.verdict h3 { font-family:'Bricolage Grotesque',sans-serif; font-size:1.8rem; font-weight:800; margin:0 0 .4rem; line-height:1.1; }
.verdict p { color: var(--muted); margin:0; line-height:1.55; }

/* Probability bar */
.tug { margin-top:1.8rem; }
.tug-head { display:flex; justify-content:space-between; font-weight:700; margin-bottom:.55rem; font-size:.92rem; }
.tug-bar { height: 12px; border-radius: 4px; background: rgba(255,255,255,.06); overflow:hidden; display:flex; }
.tug-bar .a { height:100%; transition: width .8s cubic-bezier(.2,.8,.2,1); animation: grow 1s cubic-bezier(.2,.8,.2,1); border-right:2px solid var(--panel); }
.tug-bar .b { height:100%; flex:1; }
@keyframes grow { from { width: 0 !important; } }

/* Stat chips */
.chips { display:grid; grid-template-columns: repeat(4, 1fr); gap:.8rem; margin-top:1.6rem; }
.chip { padding:.9rem 1rem; border-radius:10px; background: rgba(255,255,255,.03); border:1px solid var(--edge);
  transition: border-color .2s ease; }
.chip:hover { border-color: rgba(255,255,255,.22); }
.chip .v { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.5rem; }
.chip .l { color: var(--muted); font-size:.8rem; font-weight:600; margin-top:.15rem; }
@media (max-width: 760px) { .chips { grid-template-columns: repeat(2, 1fr); } .ring { width:190px; height:190px; } .ring .num { font-size:2.7rem; } }

/* Notices */
.notice { padding:1rem 1.2rem; border-radius:10px; border:1px solid var(--edge); background: rgba(255,255,255,.04); line-height:1.5; }
.notice.warn { border-color: rgba(227,179,65,.4); background: rgba(227,179,65,.06); }
.notice.win  { border-color: rgba(80,200,140,.4); background: rgba(80,200,140,.06); }
.notice b { font-family:'Bricolage Grotesque',sans-serif; font-size:1.1rem; }

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
      #cb-orb { position: fixed; top: 0; left: 0; pointer-events: none; z-index: 99999; border-radius: 50%;
        width: 30px; height: 30px; margin: -15px 0 0 -15px; border: 1.5px solid rgba(255,255,255,.55);
        transition: width .2s, height .2s, margin .2s, border-color .2s; }
      #cb-orb.hot { width: 46px; height: 46px; margin: -23px 0 0 -23px; border-color: rgba(232,73,127,.9); }
      .cb-burst { position: fixed; pointer-events: none; z-index: 99998; width: 10px; height: 10px; margin: -5px 0 0 -5px;
        border-radius: 50%; border: 1.5px solid rgba(255,255,255,.6); animation: cbb .6s ease-out forwards; }
      @keyframes cbb { to { transform: scale(5); opacity: 0; } }
    `;
    doc.head.appendChild(st);
    const orb = doc.createElement("div"); orb.id = "cb-orb";
    doc.body.appendChild(orb);
    let x = -100, y = -100, ox = x, oy = y;
    doc.addEventListener("mousemove", e => { x = e.clientX; y = e.clientY; });
    doc.addEventListener("mouseover", e => {
      const hot = e.target.closest && e.target.closest('[data-baseweb="select"], [role="slider"], input, button, .chip');
      orb.classList.toggle("hot", !!hot);
    });
    doc.addEventListener("mousedown", e => {
      const b = doc.createElement("div"); b.className = "cb-burst";
      b.style.left = e.clientX + "px"; b.style.top = e.clientY + "px";
      doc.body.appendChild(b); setTimeout(() => b.remove(), 650);
    });
    (function loop() {
      ox += (x - ox) * 0.25; oy += (y - oy) * 0.25;
      orb.style.transform = `translate(${ox}px, ${oy}px)`;
      requestAnimationFrame(loop);
    })();
  } catch (err) { /* cursor effects are optional */ }
})();
</script>
"""


def render_result(batting_team, bowling_team, chasing_pct, bowling_pct, feats, balls_bowled, current_score, wickets_lost):
    color = CHASE_COLOR
    bowl_color = DEFEND_COLOR

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
            <div class="a" style="width:{chasing_pct:.2f}%; background:{color}"></div>
            <div class="b" style="background:{bowl_color}"></div>
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