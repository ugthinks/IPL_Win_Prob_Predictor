import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "models" / "logistic_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- Main page ---------- */

    .stApp {
        background-color: #f7f8fa;
        color: #202124;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 45px;
        padding-bottom: 60px;
    }


    /* ---------- Header ---------- */

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        color: #202124;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 16px;
        margin-bottom: 38px;
    }


    /* ---------- Input section ---------- */

    .section-title {
        font-size: 20px;
        font-weight: 650;
        color: #202124;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .input-box {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 14px;
        padding: 28px 30px 20px 30px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }


    /* ---------- Labels ---------- */

    label {
        color: #374151 !important;
        font-weight: 500 !important;
    }


    /* ---------- Predict button ---------- */

    div.stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 9px;
        border: none;
        background-color: #e63946;
        color: white;
        font-size: 16px;
        font-weight: 600;
        margin-top: 14px;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        background-color: #c92f3b;
        color: white;
    }


    /* ---------- Result ---------- */

    .result-box {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 14px;
        padding: 28px 35px;
        margin-top: 28px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .result-title {
        text-align: center;
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 22px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .team-name {
        font-size: 19px;
        font-weight: 600;
        color: #202124;
    }

    .probability {
        font-size: 24px;
        font-weight: 700;
        color: #202124;
        text-align: right;
    }

    .result-row {
        padding: 16px 4px;
        border-bottom: 1px solid #eeeeee;
    }

    .result-row:last-child {
        border-bottom: none;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 40px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏏 IPL Match Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the winning probability from the current match situation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TEAMS
# ============================================================

teams = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad"
]


# ============================================================
# CITIES
# ============================================================

cities = [
    "Ahmedabad",
    "Bengaluru",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Jaipur",
    "Kolkata",
    "Lucknow",
    "Mumbai",
    "Pune",
    "Mohali",
    "Chandigarh",
    "Dharamsala",
    "Visakhapatnam",
    "Dubai",
    "Abu Dhabi",
    "Sharjah"
]


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="input-box">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Match Situation</div>',
    unsafe_allow_html=True
)


# First row
col1, col2, col3 = st.columns(3)

with col1:
    batting_team = st.selectbox(
        "Chasing Team",
        teams
    )

with col2:
    bowling_team = st.selectbox(
        "Bowling Team",
        teams,
        index=1
    )

with col3:
    city = st.selectbox(
        "Venue City",
        cities
    )


# Second row
col1, col2, col3 = st.columns(3)

with col1:
    target_score = st.number_input(
        "Target Score",
        min_value=1,
        max_value=400,
        value=190,
        step=1
    )

with col2:
    current_score = st.number_input(
        "Current Score",
        min_value=0,
        max_value=400,
        value=100,
        step=1
    )

with col3:
    wickets_lost = st.number_input(
        "Wickets Lost",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )


# Third row
col1, col2, col3 = st.columns(3)

with col1:
    overs_completed = st.number_input(
        "Overs Completed",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        step=0.1,
        format="%.1f"
    )


with col2:
    st.write("")

    predict_button = st.button(
        "Predict Win Probability",
        use_container_width=True
    )

with col3:
    st.write("")


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if batting_team == bowling_team:

        st.error("Chasing team and bowling team must be different.")

        st.stop()

    if current_score > target_score:

        st.error("Current score cannot be greater than the target.")

        st.stop()

    if wickets_lost >= 10:

        st.error("The batting team has lost all 10 wickets.")

        st.stop()


    # --------------------------------------------------------
    # CRICKET OVERS CALCULATION
    # --------------------------------------------------------
    #
    # Example:
    # 12.0 = 12 overs = 72 balls
    # 12.3 = 12 overs + 3 balls = 75 balls
    #
    # DO NOT treat 12.3 as 12.3 normal decimal overs.
    # --------------------------------------------------------

    completed_overs = int(overs_completed)
    completed_balls = int(
        round((overs_completed - completed_overs) * 10)
    )

    # Prevent invalid cricket notation such as 12.7
    if completed_balls > 5:

        st.error(
            "Invalid overs format. Use cricket notation such as 12.3."
        )

        st.stop()


    legal_balls = (
        completed_overs * 6
        + completed_balls
    )


    total_match_balls = 120

    balls_left = (
        total_match_balls - legal_balls
    )

    runs_left = (
        target_score - current_score
    )

    wickets_left = (
        10 - wickets_lost
    )


    # --------------------------------------------------------
    # HANDLE COMPLETED MATCH SITUATIONS
    # --------------------------------------------------------

    if runs_left <= 0:

        batting_probability = 1.0
        bowling_probability = 0.0

    elif balls_left <= 0:

        batting_probability = 0.0
        bowling_probability = 1.0

    elif wickets_left <= 0:

        batting_probability = 0.0
        bowling_probability = 1.0

    else:

        # ----------------------------------------------------
        # CURRENT RUN RATE
        # ----------------------------------------------------

        if legal_balls > 0:

            crr = (
                current_score * 6
                / legal_balls
            )

        else:

            crr = 0.0


        # ----------------------------------------------------
        # REQUIRED RUN RATE
        # ----------------------------------------------------

        rrr = (
            runs_left * 6
            / balls_left
        )


        # ----------------------------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "batting_team": [
                batting_team
            ],

            "bowling_team": [
                bowling_team
            ],

            "city": [
                city
            ],

            "runs_left": [
                runs_left
            ],

            "balls_left": [
                balls_left
            ],

            "wickets_left": [
                wickets_left
            ],

            "total_runs_x": [
                target_score
            ],

            "crr": [
                crr
            ],

            "rrr": [
                rrr
            ]
        })


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_data
        )[0]

        bowling_probability = probabilities[0]

        batting_probability = probabilities[1]


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-title">Win Probability</div>',
        unsafe_allow_html=True
    )


    # Batting team
    st.markdown(
        f"""
        <div class="result-row">
            <div class="team-name">
                {batting_team}
            </div>
            <div class="probability">
                {batting_probability * 100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Bowling team
    st.markdown(
        f"""
        <div class="result-row">
            <div class="team-name">
                {bowling_team}
            </div>
            <div class="probability">
                {bowling_probability * 100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">IPL Match Predictor · Logistic Regression</div>',
    unsafe_allow_html=True
)