from ast import Param
from enum import unique
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

classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN','SVM','Random Forest')
)
# ab hum na ak function define karna hai dataset ko load karna ka liya
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y 
    # ab function ko bula lay gayn or X .y Variable k equal rakh layn gay

X , y = get_dataset(dataset_name)
# Now we will show the shape of the data and print the unique values on the app.
st.write('Shape of dataset :', X.shape)
st.write('Number of Classes : ', len(np.unique(y)))  
# Next hum different classifier k perameter ko user input may add karien gay
def add_parameter_ui(classifier_name):
    params = dict()     #create empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C']=C # degree of correct classification
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K # its the number of Nerest Neighbour
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth # depth of every tree that grow in the random forest
        n_estimators = st.select_slider('n_estimators' , 1 ,100)
        params['n_estimators'] = n_estimators   # number of tress

    return params

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name,params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],random_state=1234)
    return clf
        
clf = get_classifier(classifier_name,params)

#
X_train,X_test,y_Train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

clf.fit(X_train,y_Train)
y_pred = clf.predict(X_test)

# model accuracy Score for print

acc  = accuracy_score(y_test,y_pred)
st.write(f'classifier= {classifier_name}')
st.write(f'Accuracy = ',acc)

### Plot Dataset ###
pca = PCA(2)
x_projected = pca.fit_transform(X)

# ab ham data o and 1 ma slice kar ka da ga

x1 = x_projected[:, 0 ]
x2 = x_projected[:, 1 ]

fig = plt.figure()
plt.scatter(x1,x2,
        c=y,alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Components 1')
plt.ylabel('Principal Components 2')

# Plot.Show()
st.pyplot(fig)