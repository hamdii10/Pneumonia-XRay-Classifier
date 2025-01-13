import streamlit as st
import numpy as np
from PIL import Image,ImageOps


        
model = tf.keras.models.load_model("X:/project 3/artifacts/model.keras")
dict = {0: 'NORMAL', 1: 'PNEUMONIA'}
image = st.file_uploader("Upload a X-ray")

if image is not None :
    st.image(image)
    image = Image.open(image)
    if len(np.array(image).shape) != 2:
        image = ImageOps.grayscale(image)
        st.warning("Invalid Image Format, converting to Grayscale")

button = st.button("Predict")



if button == True and image is not None:
    ### resize image
    image = image.resize([224, 224])
    ### Normalize image
    image = np.array(image)/255.0
    ### Predicts
    pred = model.predict(np.expand_dims(image, axis=0)      )
    if pred > 0.5:
        st.text(f"Prediction: {dict[1]}")
    else:
        st.text(f"Prediction: {dict[0]}")
