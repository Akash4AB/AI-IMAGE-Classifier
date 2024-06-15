import numpy as np
import streamlit as st
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

page_bg_css = r"""
<style>
    .stApp {
        background-image: url("https://th.bing.com/th/id/R.1a24d211c1418ceb02eb6c2c21776423?rik=B9FqWvVwcD0THg&riu=http%3a%2f%2fn.sinaimg.cn%2fsinakd20220822s%2f107%2fw2048h1259%2f20220822%2f1e23-fe43255b3e17ba079112f12eaddfe3b1.jpg&ehk=X9UUXDHqoihZuSiULTnvhn0wiVfY9pMg5AXopzhzArs%3d&risl=&pid=ImgRaw&r=0");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp > div > div:first-child {
        background: rgba(255, 255, 255, 0.7);  /* Add a translucent background for better readability */
        border-radius: 10px;
        padding: 20px;
    }
</style>
"""

# Loading the Model
model = load_model('model.h5')

def Prediction(image):
    image = Image.open(image).convert('RGB')  # Convert image to RGB format
    img = np.array(image)  # Converting image to numpy array

    test_img = cv2.resize(img, (64, 64))  # Resize image to match the model input
    test_input = test_img.reshape((1, 64, 64, 3))  # Reshape for the model
    predict = model.predict(test_input) > 0.5  # Make prediction

    prob = model.predict(test_input)  # Probability

    if predict:
        return "Real Image", prob
    else:
        return "AI Generated Image", 1 - prob

# Main function for the Streamlit app
def main():
    st.markdown(page_bg_css, unsafe_allow_html=True)

    # Title for streamlit app
    st.title('AI-IMAGE-CLASSIFIER')

    # Image uploading for input
    img_file = st.file_uploader("Upload an Image to determine if it is AI generated or REAL", type=["jpg", "jpeg", "png"])

    result = ''
    prob = None

    # Image in streamlit app
    with st.container():
        if img_file is not None:
            image = Image.open(img_file)
            image = image.resize((200, 200))
            st.image(image, caption='Uploaded Image', use_column_width=False)
        
    # Button to check result
    if st.button("Check Result"):
        if img_file is not None:
            result, prob = Prediction(img_file)
            prob = f"{prob[0][0] * 100:.2f}%"
        else:
            st.text("Image Not Found")

    # Display results with inline styles in the same row
    if result and prob is not None:
        color = "yellow"
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<h3 style="color: {color};">Type: {result}</h3>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<h3 style="color: yellow;">Probability: {prob}</h3>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()




if __name__ == '__main__':
    main()

