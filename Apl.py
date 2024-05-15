import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model
import time

# Load the trained model
model = load_model('alzheimer_cnn_modelMAIN.h5', compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the class names
class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
class_indices = {0: 'NonDemented', 1: 'VeryMildDemented', 2: 'MildDemented', 3: 'ModerateDemented'}

# Create a function to make predictions
def predict(image):
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (176, 176))  # Adjust the size according to your model input size
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_indices[predicted_class_index]
    #confidence = round(100 * (predictions[0][predicted_class_index]), 1)
    return predicted_class

# Create the Streamlit app
st.title('Alzheimer\'s Disease Detection')

# CSS Styling
st.markdown("""
    <style>
        .prediction-container {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .prediction-label {
            font-size: 24px;
            color: #2ecc71; /* Light Green */
            margin-bottom: 10px;
        }
        .center {
            display: flex;
            justify-content: center;
        }
        .spinner-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }
        .spinner {
            border: 8px solid #f3f3f3; /* Light grey */
            border-top: 8px solid #3498db; /* Blue */
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    st.write("<div class='center'>", unsafe_allow_html=True)
    st.image(image, caption='Uploaded Image.', width=300)  # Set the width to 300 pixels
    st.write("</div>", unsafe_allow_html=True)
    st.write("")
    with st.spinner(text='Classifying...'):
        time.sleep(2)  # Simulate some processing time
        label = predict(image)
        st.write("""
            <div class='prediction-container'>
                <div class='prediction-label'>Prediction: {}</div>
            </div>
        """.format(label), unsafe_allow_html=True)
