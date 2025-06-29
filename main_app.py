# Importing required libraries
import numpy as np
import streamlit as st
import cv2
import tensorflow_hub as hub
import os
from tensorflow.keras.models import load_model
from PIL import Image

# Model path (yereldeki model dosyası)
model_path = 'final_model.h5'

# Check if model file exists
if not os.path.exists(model_path):
    st.error("Model file not found! Please make sure 'final_model.h5' is in the project directory.")
    st.stop()

# Load model
model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# Defining class names
CLASS_NAMES = ['boston_bull', 'dingo', 'pekinese', 'bluetick', 'golden_retriever', 'bedlington_terrier', 'borzoi', 'basenji', 'scottish_deerhound', 'shetland_sheepdog', 'walker_hound', 'maltese_dog', 'norfolk_terrier', 'african_hunting_dog', 'wire-haired_fox_terrier', 'redbone', 'lakeland_terrier', 'boxer', 'doberman', 'otterhound', 'standard_schnauzer', 'irish_water_spaniel', 'black-and-tan_coonhound', 'cairn', 'affenpinscher', 'labrador_retriever', 'ibizan_hound', 'english_setter', 'weimaraner', 'giant_schnauzer', 'groenendael', 'dhole', 'toy_poodle', 'border_terrier', 'tibetan_terrier', 'norwegian_elkhound', 'shih-tzu', 'irish_terrier', 'kuvasz', 'german_shepherd', 'greater_swiss_mountain_dog', 'basset', 'australian_terrier', 'schipperke', 'rhodesian_ridgeback', 'irish_setter', 'appenzeller', 'bloodhound', 'samoyed', 'miniature_schnauzer', 'brittany_spaniel', 'kelpie', 'papillon', 'border_collie', 'entlebucher', 'collie', 'malamute', 'welsh_springer_spaniel', 'chihuahua', 'saluki', 'pug', 'malinois', 'komondor', 'airedale', 'leonberg', 'mexican_hairless', 'bull_mastiff', 'bernese_mountain_dog', 'american_staffordshire_terrier', 'lhasa', 'cardigan', 'italian_greyhound', 'clumber', 'scotch_terrier', 'afghan_hound', 'old_english_sheepdog', 'saint_bernard', 'miniature_pinscher', 'eskimo_dog', 'irish_wolfhound', 'brabancon_griffon', 'toy_terrier', 'chow', 'flat-coated_retriever', 'norwich_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'english_foxhound', 'gordon_setter', 'siberian_husky', 'newfoundland', 'briard', 'chesapeake_bay_retriever', 'dandie_dinmont', 'great_pyrenees', 'beagle', 'vizsla', 'west_highland_white_terrier', 'kerry_blue_terrier', 'whippet', 'sealyham_terrier', 'standard_poodle', 'keeshond', 'japanese_spaniel', 'miniature_poodle', 'pomeranian', 'curly-coated_retriever', 'yorkshire_terrier', 'pembroke', 'great_dane', 'blenheim_spaniel', 'silky_terrier', 'sussex_spaniel', 'german_short-haired_pointer', 'french_bulldog', 'bouvier_des_flandres', 'tibetan_mastiff', 'english_springer', 'cocker_spaniel', 'rottweiler']

CLASS_NAMES.sort()

# Streamlit app UI
st.title("Canine Classifier 🐶")
st.markdown("Upload a picture of a dog, and the app will predict its breed!")

# Upload image
dog_image = st.file_uploader("Upload a dog image:", type=["jpg", "jpeg", "png"])
submit = st.button("Predict")
from PIL import Image
import io
from PIL import Image
import io

if submit:
    if dog_image is not None:
        # Dosyayı byte olarak oku
        image_bytes = dog_image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Streamlit'te resmi göster
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # OpenCV formatına çevir
        opencv_image = np.array(image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        # Resize & normalize
        resized = cv2.resize(opencv_image, (350, 350))
        normalized = resized / 255.0
        input_array = np.expand_dims(normalized, axis=0)

        # Tahmin
        Y_pred = model.predict(input_array)
        predicted_label = CLASS_NAMES[np.argmax(Y_pred)]

        st.success(f"The dog breed is most likely a **{predicted_label.replace('_', ' ').title()}**.")
    else:
        st.warning("Please upload an image file first.")
