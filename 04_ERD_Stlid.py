import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport, profile_report
from streamlit_pandas_profiling import st_profile_report

# webapp ka title 
st.markdown('''
# ** Exploratitory Data Analysis Web application**
 This app is Developed by the codanics Youtube Channel Called ** EDA App ** 
 ''')

# how to upload a file from pc
with st.sidebar.header(" Upload your Dataset (.csv)"):
    upload_file = st.sidebar.file_uploader("Upload your file", type=['csv'])
    delaney_with_descriptors_url = 'https://github.com/dataprofessor/data/blob/master/delaney_solubility_with_descriptors.csv'
    df = sns.load_dataset(delaney_with_descriptors_url)
    #df = sns.load_dataset('titanic')
    #st.sidebar.markdown("[Example CSV file])(https://raw.github.com/dataprofessor/data/blob/master/delaney_solubility_with_descriptors.csvhttps://github.com/dataprofessor/data/blob/master/delaney_solubility_with_descriptors.csv)")

#profilimg report for pandas
if upload_file is not None:
    @st.cache

    def load_csv():
        csv= pd.read_csv(upload_file)
        return csv
    df= load_csv()
    pr= ProfileReport(df,explorative = True)
    st.header('***input DF***')
    st.write(df)
    st.write('---')
    st.header('***Profiling Report With Pandas***')
    st_profile_report(pr)
else:
    st.info('Awaitng for CSV file,Upload kar bi do ap ya kam nahi Lena?')
    if st.button('Press to use example data set'):
        # example dataset
        @st.cache

        def load_data():
            a = pd.DataFrame(np.random.rand(100,5),
                                columns=['age','banana','codanics','pakistan','Ear'])
            return a
        df= load_data()
        pr= profile_report(df,explorative = True)
        st.header('***input DF***')
        st.write(df)
        st.write('---')
        st.header('***Profiling Report With Pandas***')
        st_profile_report(pr)