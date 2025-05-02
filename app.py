import streamlit as st
import os
import pickle
from PIL import Image

# Load model and encoder
with open("iris_model.pkl", "rb") as f:
    model, le = pickle.load(f)

st.title("🌸 Iris Flower Classifier")

# Input sliders
sl = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width", 2.0, 4.5, 3.5)
pl = st.slider("Petal Length", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    input_data = [[sl, sw, pl, pw]]
    pred = model.predict(input_data)
    species = le.inverse_transform(pred)[0]
    st.success(f"Predicted Species: {species}")

    image_path = f"{species}.jpg"
    st.write("Image path:", image_path)
    st.write("Image exists:", os.path.exists(image_path))

    try:
        image = Image.open(image_path)
        st.image(image, caption=f"Species: {species}", use_column_width=True)
    except FileNotFoundError:
        st.error(f"Image for species '{species}' not found.")

