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
    Bat_team = st.sidebar.selectbox('Bat_team',teams)

    if Bat_team == 'AUS':
        temp_array = temp_array + [0,0,0,0,0,0,0,0]
    elif Bat_team == 'BAN':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif Bat_team == 'ENG':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif Bat_team == 'IND':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif Bat_team == 'NZ':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif Bat_team == 'PAK':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif Bat_team == 'SA':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif Bat_team == 'SL':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif Bat_team == 'WI':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]

    # Bowl_team = ''
    # teams.remove(Bat_team)
    # print(teams)
    Bowl_team  = st.sidebar.selectbox('Bowl_team',teams)
    
    if Bowl_team == 'AUS':
        temp_array = temp_array + [0,0,0,0,0,0,0,0]
    elif Bowl_team == 'BAN':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif Bowl_team == 'ENG':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif Bowl_team == 'IND':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif Bowl_team == 'NZ':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif Bowl_team == 'PAK':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif Bowl_team == 'SA':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif Bowl_team == 'SL':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif Bowl_team == 'WI':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]

    over = st.sidebar.slider('over', 0,19,10)
    ball = st.sidebar.slider('ball', 1,6,3)
    overs = float(str(over)+ '.'+ str(ball))
    total_score = st.sidebar.slider('total_score', 10,250,87)
    total_wickets = st.sidebar.slider('total_wickets', 0,9,2)
    prev_runs_30 = st.sidebar.slider('prev_runs_30', 10,90,39)
    prev_wickets_30 = st.sidebar.slider('prev_wickets_30', 0,9,1)
    prev_30_dot_balls = st.sidebar.slider('prev_30_dot_balls', 0,24,9)
    prev_30_boundaries = st.sidebar.slider('prev_30_boundaries', 0,24,3)
    temp_array = temp_array + [overs, total_score, total_wickets, prev_runs_30, prev_wickets_30, prev_30_dot_balls, prev_30_boundaries]
    data = np.array([temp_array])

    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Displays the user input features
st.subheader('User Input features')

# Reads in saved classification model
load_clf = pickle.load(open('lr-model all.pkl', 'rb'))

# Apply model to make predictions
df = np.array(input_df)
prediction = load_clf.predict(df)

st.subheader('Predicted score')
st.write(int(prediction))