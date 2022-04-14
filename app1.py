from sklearn.multioutput import ClassifierChain
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.write('''
# Exploring different Machine Learning Models and Datasets
 ''')  

 # dataset k  ame ak box may dal ka sider ma pay laga do

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris','Brest Cancer','Wine')
)

Classifier_name = st.sidebar.selectbox(
    'Select Classifier'
    ('KNN','SVM','Random Forest')
)