import streamlit as st
from PIL import Image
#import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

def preprocess_image(image):
    #image = cv2.resize(image, (224, 224))  # Resize the image to the input size expected by MobileNet
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    #image = tf.expand_dims(image, axis=0)  # Add a batch dimension
    return image

def make_prediction(image, model):
    image = preprocess_image(image)
    prediction = model.predict(image)
    return prediction

def main():
    st.title("Animal Classifier")

    st.write('Predictable classes:')
    st.write("- dog")
    st.write("- horse")
    st.write("- elephant")
    st.write("- butterfly")
    st.write("- chicken")
    st.write("- cat")
    st.write("- cow")
    st.write("- sheep")
    st.write("- squirrel")
    st.write("- spider")

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    classes = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        res_image = image.resize((224, 224))
        colored_image = res_image.convert('RGB')
        #image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = np.array(colored_image)
        image = np.expand_dims(image, axis=0)
        print(image.shape)
        #image = cv2.resize(image, (224, 224))
        

        print(image)

        # Make predictions
        model = load_model('model1_mobile.h5')
        #processed_image = preprocess_image(image)
        prediction = make_prediction(image, model)

        st.write("Prediction:")
        st.write(classes[np.argmax(prediction)])

        
if __name__ == "__main__":
    main()
