import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score 



# make containers

header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title(" Kashti ki app")
    st.text(" in the project we eill work on kashti data")

with datasets:
    st.header("kashti doob gaye,hawwwwww")
    st.text(" in the project we will work on kashti data")
    # import data
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))

    st.subheader("Male and Female kitna admi thaaa")
    st.bar_chart(df['sex'].value_counts ())

    # other plot
    st.subheader("Class ka Hisab sa Faraq")
    st.bar_chart(df['class'].value_counts())

    # bar chart
    st.bar_chart(df['age'].sample(10))

with features:
    st.header("thse are our app features")
    st.text(" in the project we will work on kashti data")
    st.markdown('1. **Feature 1:** This will tell us pata ni ')
    st.markdown('2. **Feature 2:** This will tell us pata ni ')
    st.markdown('3. **Feature 3:** This will tell us pata ni ')
    st.markdown('4. **Feature 4:** This will tell us pata ni ')

with model_training:
    st.header("kashti walo ka kia hova ? Model Training")
    st.text(" In this Project we will work on kashti data.....")
    # ranking column
    input, display = st.columns(2)

    # pehlay column ain ap k selection points hun
    max_depth = input.slider('How many people do you know? ', min_value = 10, max_value = 100, value =20 , step =5)



# n_estimators
n_estimators = input.selectbox("how many tree should be there in RF?",options = [50,100,200,300,'No Limit'])

# adding list of features
input.write(df.columns)

#input features from user
input_features= input.text_input('Which feature we should use')

# machine Learning Model

model  = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
# condition 
if n_estimators == 'No Limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators  )


# define x and Y
X = df[[input_features]]
Y = df[['fare']]
model.fit(X,Y)
pred = model.predict(Y)

# Display Matrics

display.subheader("Mean absolute error of the model is : ")
display.write(mean_absolute_error(Y,pred))
display.subheader("Mean Squred error of the model is : ")
display.write(mean_squared_error(Y,pred))
display.subheader("R Squred    error of the model is : ")
display.write(r2_score(Y  ,pred))
