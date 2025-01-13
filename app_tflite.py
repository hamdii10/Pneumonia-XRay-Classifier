import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image,ImageOps

    
interpreter = tf.lite.Interpreter(model_path='X:/project 3/Pneumonial/tflite_model.tflite')
interpreter.allocate_tensors()

output= interpreter.get_output_details()[0] 
input = interpreter.get_input_details()[0] 

image = st.file_uploader("Upload a X-ray")
button = st.button("Predict")

if image is not None :
    st.image(image)
    image = Image.open(image)
    if len(np.array(image).shape) != 2:
        image = ImageOps.grayscale(image)
        st.warning("Invalid Image Format, converting to Grayscale")
        
if button == True and image is not None:
    ### resize image
    image = image.resize([224, 224])
    ### Normalize image
    image = np.array(image,dtype=np.float32)/255.0
    ### Predicts
    
    input_data = np.expand_dims(image,axis=-1)
    
    interpreter.set_tensor (input['index'], np.expand_dims(input_data,axis=0))
    interpreter.invoke()
    pred = interpreter.get_tensor (output['index'])
    if pred > 0.5:
        st.text(f"Prediction: {dict[1]}")
    else:
        st.text(f"Prediction: {dict[0]}")
