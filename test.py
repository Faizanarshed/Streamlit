import streamlit as st
import seaborn as sns



st.header("this Vedio is brought to you y baba ammar  ")
st.text("kia aplogo ko is ma maza ah raha ha")
 
st.header("pata ni is ma kia likhna ha")
st.text("is ma sa kia ho raha ha pata ni baba g ")

df  = sns.load_dataset('iris')
st.write(df[['species','sepal_length','petal_length']].head(10 ))

st.bar_chart(df['sepal_length'])
st.line_chart(df['petal_length'])