import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and display the uploaded image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to apply Gaussian Blur filter
def apply_gaussian_blur(image, kernel_size):
    image = np.array(image)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

# Function to apply Sobel edge detection filter
def apply_sobel_edge_detection(image):
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(np.absolute(sobel_edges))

# Streamlit app UI
def main():
    st.title("Interactive Image Filtering and Edge Detection App")

    # File uploader to upload the image
    uploaded_image = st.file_uploader("/content/Golden+Retrievers+dans+pet+care.jpeg", type="jpg, png, jpeg")

    if uploaded_image is not None:
        # Load and display the image
        image = load_image(uploaded_image)
        st.image(image, caption="/content/Golden+Retrievers+dans+pet+care.jpeg", use_column_width=True)

        # Choose filter or edge detection technique
        option = st.selectbox(
            "Choose a filtering or edge detection technique",
            ["None", "Gaussian Blur", "Sobel Edge Detection"]
        )

        if option == "Gaussian Blur":
            kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2)
            result_image = apply_gaussian_blur(image, kernel_size)
            st.image(result_image, caption="Gaussian Blurred Image", use_column_width=True)

        elif option == "Sobel Edge Detection":
            result_image = apply_sobel_edge_detection(image)
            st.image(result_image, caption="Edge Detection (Sobel)", use_column_width=True)

        elif option == "None":
            st.write("Please select a filter or edge detection technique.")

    else:
        st.write("Please upload an image to start.")

if __name__ == "__main__":
    main()

