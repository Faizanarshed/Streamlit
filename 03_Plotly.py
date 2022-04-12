#import Library
from turtle import width
import pandas as pd
import streamlit as st
import plotly.express as px

# import dataset
st.title("Plotly and Streamlit ko mila ka app bananai ha")
df = px.data.gapminder()
st.write(df)
# st.write(df.head())
st.write(df.columns)
# sumary stats

st.write(df.describe())

# data manegement according to plotly
year_option = df['year'].unique().tolist()
year = st.selectbox('which year should be plot',year_option,0)
#df = df[df['year']==year]  

# plotly
fig = px.scatter(df,x='gdpPercap',y = 'lifeExp',size = 'pop',color = 'continent',hover_name='continent',
                log_x=True,size_max=55,range_x=[100,100000],range_y=[20,90],
                animation_frame='year',animation_group='country')

fig.update_layout(width = 800,height = 400)


st.write(fig)
