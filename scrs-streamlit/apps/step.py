import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import urllib
import codecs
import streamlit.components.v1 as component
from .function_file import *

def step():
    choices = st.sidebar.radio("", ('Overview', 'Preprocessing Steps', 'Comparison', 'Real-Time Detection'))

    if choices == 'Overview':
        st.header("Overview")
        """
        This app is to show all the steps of preparing the **Dataset** & **Model**.
        Basic Idea is to develop it, is that to show all the steps that are involved in  building Model.
        \n First, we discuss about the **original dataset**: \n
        There are **Eight** Classes of Diseases named as: \n
        **1-**   Melanoma \n
        **2-**   Benign Keratosis-like Lesions \n
        **3-**   Basal Cell Carcinoma \n
        **4-**   Actinic Keratoses and Intraepithelial Carcinomae \n
        **5-**   Dermatofibroma \n
        **6-**   Pyogenic Granulomas and Hemorrhage \n
        **7-**   Insect Bite \n
        **8-**   Acne \n

        Now, take a look at **Preparing Dataset's** Steps:
        * Resizing All Images to 224x224.
        * All classes must have 1000 Training, 250 Testing Images
        * All Images should not have noise like Hair etc. \n
        Now, go to Steps to show implementation.
        """

    elif choices == 'Preprocessing Steps':
        st.header("Preprocessing Steps:-")
        ch = st.radio("Implementation Steps",('Image Resizing','Hair Removal',
        'Background Removal', 'Body Extraction', 'Detect Faces', "Data Augmentation", 'Segmentation'))
        if ch == 'Image Resizing':
            st.markdown('<br><br>', unsafe_allow_html=True)
            """
            **Image Resizing:**
            \n This process convert the image into 224x224. This process helps in the Extraction of pixels
            in the Image. \n
            """
            st.subheader("Live Demo:- ")
            uploaded_file = st.file_uploader(
                "Choose an image to classify", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                bytes_data = uploaded_file.read()
                st.write("Original Image")
                st.image(uploaded_file)
                image = Image.open(BytesIO(bytes_data))
                image = np.array(image)
                open_cv_image = image[:, :, ::-1].copy()
                st.write(open_cv_image.shape)
                res_img = resize_img(open_cv_image)
                st.write("Output Image")
                st.image(res_img)
                st.write(res_img.shape)

        elif ch == 'Hair Removal':
            st.markdown('<br><br>', unsafe_allow_html=True)
            """
            **Hair Removal:**
            \n This process remove the Hair from the images. This process helps in the Extraction the features
            and Training the Image. \n
            """
            st.subheader("Live Demo:- ")
            uploaded_file = st.file_uploader(
                "Choose an image to classify", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                col1, col2 = st.beta_columns(2)
                bytes_data = uploaded_file.read()
                col1.write("Original Image")
                image = Image.open(BytesIO(bytes_data))
                image = np.array(image)
                open_cv_image = image[:, :, ::-1].copy()
                res_img = resize_img(open_cv_image)
                col1.image(res_img)
                noise_remove = hair_removal(res_img)
                col2.write("Output Image")
                col2.image(noise_remove)

        elif ch == 'Background Removal':
            st.markdown('<br><br>', unsafe_allow_html=True)
            """
            **Background Removal:**
            \n This process remove the Irrelevent Background from the images.
            This process take place when there is body in image and we have to extract only body parts. \n
            """
            st.subheader("Live Demo:- ")
            uploaded_file = st.file_uploader(
                "Choose an image to classify", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                col1, col2 = st.beta_columns(2)
                bytes_data = uploaded_file.read()
                col1.write("Original Image")
                image = Image.open(BytesIO(bytes_data))
                image = np.array(image)
                open_cv_image = image[:, :, ::-1].copy()
                res_img = resize_img(open_cv_image)
                col1.image(res_img)
                read_img(res_img, uploaded_file.name)
                br = back_removal(uploaded_file.name)

                col2.write("Background Remove Image")
                col2.image(br)

        elif ch == "Segmentation":
            st.markdown('<br><br>', unsafe_allow_html=True)
            """
            **Segmentation:**
            \n This process Extract the affected parts within the image and label a suitable color to it.
            This process detect the boudary of the diseases. \n
            """
            st.subheader("Live Demo:- ")
            uploaded_file = st.file_uploader(
                "Choose an image to classify", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                col1, col2 = st.beta_columns(2)
                bytes_data = uploaded_file.read()
                col1.write("Original Image")
                image = Image.open(BytesIO(bytes_data))
                image = np.array(image)
                open_cv_image = image[:, :, ::-1].copy()
                res_img = resize_img(open_cv_image)
                col1.image(res_img)
                seg_img = segmentation(res_img)
                col2.write("Output Image")
                col2.image(seg_img)

        elif ch == "Data Augmentation":
            st.markdown('<br><br>', unsafe_allow_html=True)
            """
            **Data Augmentation:**
            \n This process take place to increase the number of images for training. \n
            """
            st.subheader("Live Demo:- ")
            dest_folder = "augment_folder/"
            li = os.listdir(dest_folder)
            st.write("Augmentized Image")
            for i in li:
                path = os.path.join(dest_folder + i)
                img = cv2.imread(path)
                st.image(img)


    elif choices == 'Comparison':
        new_choice = st.selectbox("Comparison", ('Accuracy', 'Loss'))


    elif choices == 'Real-Time Detection':
        new_choice = st.selectbox("Real-Time Detection", ())
