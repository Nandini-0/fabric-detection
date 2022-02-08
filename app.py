import streamlit as st 
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
from keras.preprocessing.image import load_img,img_to_array
import pickle
from tensorflow.keras.preprocessing import image

st.header("Human, Mannequin and Fabric Detection")
st.text('-'*80)

m1 = tf.keras.models.load_model('human_detection_3.h5')
dic = pickle.load(open(r'class_3.pkl','rb'))

#dic=train_generator.class_indices
icd={k:v for v,k in dic.items()}
def output(Img):
    img = np.array(Img)
    #img = load_img(Img, target_size=(224, 224))
    img=img/255
    img=np.expand_dims(img,axis = 0)
    answer=m1.predict(img)
    #answer=m1.predict_classes(img)
    classes_x=np.argmax(answer)
    classes_x = icd[classes_x]
    probability=round(np.max(m1.predict(img)*100),2)
    st.write('Item Is : ',classes_x)


Img = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
if Img is None:
    st.text("Please upload an image")
else:
    st.image(Img)
    Img = Image.open(Img)
    newsize = (224, 224)
    img = Img.resize(newsize)
    #Img = image.load_img(Img, target_size=(224,224,3))
    output(img)
    #output('14.jpg')


