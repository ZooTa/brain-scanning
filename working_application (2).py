import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('my_model.h5')  # Provide the correct path to your model file

# Define class descriptions
class_descriptions = {
    0: "CNV (Choroidal Neovascularization): A condition involving the growth of abnormal blood vessels beneath the retina.",
    1: "DME (Diabetic Macular Edema): Swelling in the macula, a result of diabetes, which can lead to vision problems.",
    2: "DRUSEN: Yellow deposits under the retina that may be a sign of age-related macular degeneration.",
    3: "NORMAL: A healthy retina without any significant issues."
}

st.title("Retina Classification App")

# Upload an image for classification
uploaded_image = st.file_uploader("Upload an OCT image", type=["jpg", "jpeg"])
if uploaded_image is not None:
    # Preprocess the uploaded image
    image = tf.image.decode_jpeg(uploaded_image.read(), channels=1)
    image = tf.image.resize_with_crop_or_pad(image, target_height=192, target_width=192)
    image = tf.image.resize(image, (192, 192))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform inference
    prediction = model.predict(image)
    class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

    # Create a layout with the image on the left and predictions/description on the right
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.write("Class Prediction:")
        st.write(class_names[np.argmax(prediction)])
        
        # Display class description
        predicted_class = np.argmax(prediction)
        st.write("Class Description:")
        st.write(class_descriptions[predicted_class])
        st.write("Raw Predictions:")    
        st.write(prediction)


