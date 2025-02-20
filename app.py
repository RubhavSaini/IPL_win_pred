import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe1 = pickle.load(open('pipe.pkl','rb'))
pipe2 = pickle.load(open('pipe1.pkl','rb'))

models = {
    "Model 1 :(Logistic Regress)": pipe1,
    "Model 2 :(XGBoost Classifier)": pipe2,
}
# print(models)
# st.title("<h1 style='text-align: center; color: red;'>IPL Win Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<h1 style='text-align: center; color: red;'>IPL Win Predictor</h1>",
    unsafe_allow_html=True
)
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score',min_value=0, max_value=1000, value=1, step=1)
with col4:
    overs = st.number_input('Overs completed',min_value=0, max_value=20, value=1, step=1)
with col5:
    wickets = st.number_input('Wickets out',min_value=0, max_value=10, value=0, step=1)
# model=pipe1

model_name= st.selectbox('Select model',models)
pipe=models[model_name]
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    # st.header(batting_team + "- " + str(round(win * 100)) + "%")
    # st.header(bowling_team + "- " + str(round(loss * 100)) + "%")
    if loss>win:
        st.error(batting_team + "- " + str(round(win*100)) + "%")
        st.success(bowling_team + "- " + str(round(loss*100)) + "%")
    else:
        st.success(batting_team + "- " + str(round(win * 100)) + "%")
        st.error(bowling_team + "- " + str(round(loss * 100)) + "%")





