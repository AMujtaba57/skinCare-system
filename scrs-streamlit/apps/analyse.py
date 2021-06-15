import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import urllib
import pandas as pd
from .image_preprocessing import *
from .function_file import *


def layout():
    st.subheader("""Choose an Image:""")
    uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        bytes_data = uploaded_file.read()
        image = Image.open(BytesIO(bytes_data))
        image = np.array(image)
        open_cv_image = image[:, :, ::-1].copy()

        st.write("""
        ### Uploaded Image:
        """)
        img = resize_img(open_cv_image)
        st.write(""" Image is resized to """)
        st.write(img.shape)
        st.image(img)

        col1, col2 = st.beta_columns((2,4))
        proc = col2.button("Predict Result")
        if proc:
            st.markdown("<br><br>", unsafe_allow_html=True)
            col3, col4 =  st.beta_columns(2)
            col3.write("""
                #### Feature Extraction:
            """)
            col3.markdown("<br>", unsafe_allow_html=True)

            cv2.imwrite("./processed_img/"+uploaded_file.name, img)
            out_img = background_removal("./processed_img/"+uploaded_file.name)
            fimg = face_facial(out_img)
            col3.image(fimg)
            cv2.imwrite("./processed_img/"+'facial.jpg', img)



            col4.write("""
                #### Semantic Segmentation:
                 Effected part is highlighted
            """)
            # st.write(fimg)
            img_seg = segmentation(img)
            col4.image(img_seg)


            st.markdown("<br><br>", unsafe_allow_html=True)
            st.write("""
                #### Prediction:
            """)
            st.markdown("<br>", unsafe_allow_html=True)


            data = run_example("./processed_img/"+'facial.jpg')

            st.write(data["data"])

            string = f"Your Infection mostly related to '{data['img_name']}'"
            st.write("Result:")
            st.success(string)

            st.markdown("<br><br>", unsafe_allow_html=True)
            
            st.write("""
                #### Medication:
            """)
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.write(data["drugs"])
