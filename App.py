import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("ipl_model.pkl")

st.title("🏏 IPL Win Predictor")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select the batting team", 
                                ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
                                 "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
                                 "Delhi Capitals", "Punjab Kings"])

with col2:
    bowling_team = st.selectbox("Select the bowling team", 
                                ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
                                 "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
                                 "Delhi Capitals", "Punjab Kings"])

venue = st.text_input("Selected host city")

target = st.number_input("Target", min_value=0, step=1)
score = st.number_input("Score", min_value=0, step=1)
overs = st.number_input("Overs completed", min_value=0.0, step=0.1, format="%.1f")
wickets = st.number_input("Wickets out", min_value=0, max_value=10, step=1)

# --- Prediction Button ---
if st.button("Predict Probability"):
    # Prepare input dataframe
    input_df = pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "venue": [venue],
        "target": [target],
        "score": [score],
        "overs": [overs],
        "wickets": [wickets]
    })

    # Preprocess input_df if needed (label encoding, scaling etc.)
    # Must be the same preprocessing as your training

    # Predict probability
    try:
        win_prob = model.predict_proba(input_df)[0][1] * 100  # assuming 1st column is lose, 2nd is win
        st.subheader(f"{batting_team} - {win_prob:.0f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")