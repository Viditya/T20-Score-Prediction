import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Score Prediction App

This app predicts the score of a T20 match!

Data obtained from [cricsheet](https://cricsheet.org/downloads/).
""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe
def user_input_features():
    temp_array = list()
    teams = ['AUS','BAN','ENG','IND','NZ','PAK','SA','SL','WI']
    bat_team = st.sidebar.selectbox('Batting team', teams)

    if bat_team == 'AUS':
        temp_array = temp_array + [0,0,0,0,0,0,0,0]
    elif bat_team == 'BAN':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif bat_team == 'ENG':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif bat_team == 'IND':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif bat_team == 'NZ':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif bat_team == 'PAK':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif bat_team == 'SA':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif bat_team == 'SL':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif bat_team == 'WI':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]

    # bowl_team = ''
    # teams.remove(bat_team)
    # print(teams)
    bowl_team  = st.sidebar.selectbox('Bowling team', [team for team in teams if team != bat_team])
    
    if bowl_team == 'AUS':
        temp_array = temp_array + [0,0,0,0,0,0,0,0]
    elif bowl_team == 'BAN':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif bowl_team == 'ENG':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif bowl_team == 'IND':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif bowl_team == 'NZ':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif bowl_team == 'PAK':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif bowl_team == 'SA':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif bowl_team == 'SL':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif bowl_team == 'WI':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]

    over = st.sidebar.slider('Over no', 0,19,10)
    ball = st.sidebar.slider('Ball no', 1,6,3)
    overs = float(str(over)+ '.'+ str(ball))
    total_score = st.sidebar.slider('Total Score', 10,250,87)
    total_wickets = st.sidebar.slider('Total Wickets', 0,9,2)
    prev_runs_30 = st.sidebar.slider('Prev Runs_30', 10,90,39)
    prev_wickets_30 = st.sidebar.slider('Wickets in last 5 overs', 0,9,1)
    prev_30_dot_balls = st.sidebar.slider('Dot balls in last 5 overs', 0,24,9)
    prev_30_boundaries = st.sidebar.slider('Boundaries in last 5 overs', 0,24,3)
    temp_array = temp_array + [overs, total_score, total_wickets, prev_runs_30, prev_wickets_30, prev_30_dot_balls, prev_30_boundaries]
    data = np.array([temp_array])

    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Displays the user input features
# st.subheader('User Input features')

# Reads in saved classification model
load_clf = pickle.load(open('lr-model all.pkl', 'rb'))

# Apply model to make predictions
df = np.array(input_df)
prediction = load_clf.predict(df)

st.subheader('Predicted score')
st.subheader(int(prediction))