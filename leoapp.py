from email.mime import audio
from matplotlib import image
import streamlit as st
from PIL import Image

st.write(
    '''
    ## Add Media Files in Streamlit application

    ''')

st.write('''
    
    ## add Image into the app '''

    )
image1 =  image.open('snowleopard.jpg')
st.image(image)


st.write('''
### Add Video file in to the streamlit App
''')
video1 = open("lep.mp4","rb")
st.video(video1)

st.write('''
## Add Audio file into the streamlit library
''')
audio1= open("leo.mp3","rb")
st.audio(audio1)