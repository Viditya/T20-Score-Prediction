import pickle

from keras.models import load_model
import numpy as np
import pandas as pd
import streamlit as st

st.title("T20 Score Prediction App")
st.write("This app predicts the score of a T20 match!")
st.caption("""Data obtained from [cricsheet](https://cricsheet.org/downloads/)""")

st.subheader("User Input Features")

col1, col2 = st.columns(2)


# Collects user input features into dataframe
def user_input_features():
    temp_array = list()
    teams = ["AUS", "BAN", "ENG", "IND", "NZ", "PAK", "SA", "SL", "WI"]
    with col1:
        bat_team = st.selectbox("Batting team", teams)

    if bat_team == "AUS":
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0]
    elif bat_team == "BAN":
        temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif bat_team == "ENG":
        temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif bat_team == "IND":
        temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif bat_team == "NZ":
        temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif bat_team == "PAK":
        temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif bat_team == "SA":
        temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif bat_team == "SL":
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif bat_team == "WI":
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

    with col2:
        bowl_team = st.selectbox(
            "Bowling team", [team for team in teams if team != bat_team]
        )

    if bowl_team == "AUS":
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 0]
    elif bowl_team == "BAN":
        temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif bowl_team == "ENG":
        temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif bowl_team == "IND":
        temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif bowl_team == "NZ":
        temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif bowl_team == "PAK":
        temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif bowl_team == "SA":
        temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif bowl_team == "SL":
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif bowl_team == "WI":
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

    with col1:
        over = st.slider("Current ongoing over", 0, 19, 10)
        total_score = st.slider(f"Current Score of {bat_team}", 10, 250, 87)
        prev_runs_30 = st.slider("Runs scored in last 5 overs", 10, 90, 39)
        prev_30_dot_balls = st.slider("Dot balls in last 5 overs", 0, 24, 9)

    with col2:
        ball = st.slider(f"Balls bowled in over {over}", 1, 6, 3)
        total_wickets = st.slider("Total wickets fallen", 0, 9, 2)
        prev_wickets_30 = st.slider("Wickets in last 5 overs", 0, 9, 1)
        prev_30_boundaries = st.slider("Boundaries in last 5 overs", 0, 24, 3)

    overs = float(str(over) + "." + str(ball))
    temp_array = temp_array + [
        overs,
        total_score,
        total_wickets,
        prev_runs_30,
        prev_wickets_30,
        prev_30_dot_balls,
        prev_30_boundaries,
    ]
    data = np.array([temp_array])

    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

def run_model(selected, input):
    if selected == 'linear':
        model_path = open("lr-model all.pkl", "rb")
        # Reads in saved classification model
        load_clf = pickle.load(model_path)
        df = np.array(input)
        # Apply model to make predictions
        prediction = load_clf.predict(df)
    elif selected == 'neural':
        model_path = 'nn-model-all.h5'
        # Reads in saved neural model
        model = load_model(model_path, compile=False)
        model.make_predict_function()
        df = np.array(input)
        # Apply model to make predictions
        prediction = model.predict(df)
    return prediction

prediction = run_model('neural', input_df)

st.header("Predicted score")
st.subheader(int(prediction))
