import streamlit as st
from model import RetinaModel  # Import your RetinaModel class from model.py
import numpy as np
from PIL import Image

# Define class names and descriptions here
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
class_descriptions = {
    0: "CNV (Choroidal Neovascularization): A condition involving the growth of abnormal blood vessels beneath the retina.",
    1: "DME (Diabetic Macular Edema): Swelling in the macula, a result of diabetes, which can lead to vision problems.",
    2: "DRUSEN: Yellow deposits under the retina that may be a sign of age-related macular degeneration.",
    3: "NORMAL: A healthy retina without any significant issues."
}

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((192, 192))
    image = image.convert('RGB')  # Ensure the image has 3 channels (RGB)
    image = np.array(image) / 255.0  # Normalize the image
    return image

# Function to make predictions
def predict(image):
    image = preprocess_image(image)
    print("Preprocessed Image Shape:", image.shape)
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    prediction = model.predict(image)
    print("Raw Prediction Scores:", prediction)
    return prediction

# Streamlit UI
st.title('Retina Image Classification')
st.write('This app classifies retina images into four categories: CNV, DME, DRUSEN, NORMAL.')

# Instantiate your model
model = RetinaModel()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    if not model.model:
        st.write("Warning: Model weights not found. You should train the model first.")
    else:
        prediction = model.predict(uploaded_image)  # Call the predict method

        # Create a two-column layout
        col1, col2 = st.beta_columns([1, 2])
        with col1:
            st.image(uploaded_image, use_column_width=True)
        with col2:
            st.write("Class Prediction:")
            st.write(class_names[np.argmax(prediction)])
            st.write("Class Description:")
            st.write(class_descriptions[np.argmax(prediction)])
            st.write("Raw Predictions:")
            st.write(prediction)
            st.write("Prediction:")
            for i in range(4):
                st.write(f"{class_names[i]}: {prediction[0][i]:.2%}")
