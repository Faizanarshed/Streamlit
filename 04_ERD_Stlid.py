import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
from pandas_profiling import profile_report
from streamlit_pandas_profiling import st_profile_report

# webapp ka title 
st.markdown('''
# ** Exploratitory Data Analysis Web application**
 This app is Developed by the codanics Youtube Channel Called ** EDA App ** 
 ''')

# how to upload a file from pc
with st.sidebar(" Upload your Dataset (.CSV)"):
    upload_file = st.sidebar.file_uploader("Upload your file", type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file]")(df)