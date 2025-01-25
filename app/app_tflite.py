import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# Set the title and description
st.title("Pneumonia Detection Application")
st.write("""
This application predicts whether an uploaded chest X-ray image indicates pneumonia. 
Please upload a valid chest X-ray image and click 'Predict'.
""")

# Load the TensorFlow Lite model
model_path = os.path.join('models', 'tflite_model.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Retrieve input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("""
This pneumonia detection model uses a pre-trained deep learning network.
The model has been optimized and converted into TensorFlow Lite format for efficient inference.
""")

st.sidebar.header("How It Works")
st.sidebar.write("""
1. Upload a chest X-ray image.
2. Click the 'Predict' button to get the result.
3. Ensure the image is in grayscale or will be automatically converted.
""")

st.sidebar.header("Developer Notes")
st.sidebar.write("""
- This application is built with Python and Streamlit.
- The machine learning model was trained using TensorFlow.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["png", "jpg", "jpeg"])
predict_button = st.button("Predict")

# Prediction logic
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)
    
    # Ensure the image is grayscale
    if len(np.array(image).shape) != 2:
        image = ImageOps.grayscale(image)
        st.warning("The uploaded image was converted to grayscale.")

    if predict_button:
        # Resize image
        image = image.resize((224, 224))

        # Normalize image
        image = np.array(image, dtype=np.float32) / 255.0
        input_data = np.expand_dims(image, axis=(0, -1))

        # Perform prediction
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        # Display the result
        if prediction > 0.5:
            st.error("The model predicts this X-ray indicates *pneumonia*. Please consult a medical professional.")
        else:
            st.success("The model predicts this X-ray is *normal*.")

st.sidebar.header("Contact")
st.sidebar.write("""
For questions or suggestions, reach out to:
- **Email**: [your_email@example.com](mailto:your_email@example.com)
- **GitHub**: [your_github](https://github.com/your_github)
""")
