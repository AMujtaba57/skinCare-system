import os
import cv2
import random as random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
from pixellib.tune_bg import alter_bg
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model,model_from_json
import json
import pandas as pd
from .treatment import *

def resize_img(img):
    if img.shape[0] == 224 and img.shape[1] == 224:
        return img
    else:
        img_resize = cv2.resize(img, (224,224))
        # img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        return img_resize


def hair_removal(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,thresh2,1,cv2.INPAINT_TELEA)
    return dst

def read_img(img_data, img_name):
    if img_data != '':
        cv2.imwrite(img_name, img_data)

def back_removal(img):
    change_bg = alter_bg()
    change_bg.load_pascalvoc_model("./trained_models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    output = change_bg.color_bg(img, colors = (0, 128, 0))
    return output

def segmentation(img):
    print(img)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, thresh =     cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    kernel = np.ones((6,6),np.uint8)
    dilate = cv2.dilate(opening,kernel,iterations=3)
    blur = cv2.blur(dilate,(15,15))
    ret, thresh =     cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy =     cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask)
    coloured = res.copy()
    coloured[mask == 255] = (0, 0, 255)
    dst = cv2.addWeighted(img, 0.6, coloured, 0.4, 0)
    return dst

def augmentation():
    dest_folder = "augment_folder/"
    li = os.listdir(dest_folder)
    for i in li:
        img = cv2.imread(os.path.join(dest_folder + "i"))
        return img

def simple_classifier(img):
    model = tf.keras.models.load_model("./trained_models/model1.h5", custom_objects=None, compile=True, options=None)
    im_pil = Image.fromarray(img)
    im_np = np.asarray(im_pil)
    img_reshape = im_np[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


# load and prepare the image
def load_image(filename):
	# load the image
    try:
        img = load_img(filename, grayscale=False, target_size=(224,224,3))
        print(img)
        # convert to array

        img = img_to_array(img)
        # reshape into a single sample with 1 channel
        img = img.reshape(-1,224,224,3)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img
    except Exception as e:
        print("Error1: ",e)

# load an image and predict the class
def run_example(filename):

    try:
        try:
           # load the image
            img = load_image(filename)

            # load model
        except Exception as e:
                print("Error3: ",e)


        try:

            model = load_model("./trained_models/acne1.h5")
            model2=load_model("./trained_models/model1.h5")
                # predict the class
            try:
                digit = model.predict(img)
                digit2=model2.predict(img)


                dg=['Actinic','Melonoma','Basal','Acne',
                'Pigmented Benign','Insect Bites']
                cls= np.argmax(digit, axis = 1)
                cls2= np.argmax(digit2, axis = 1)
                predict=[]
                if cls2 == 0:
                    predict.append("Actinic")
                    print("Actinic Detected")
                elif cls2== 1:
                    predict.append("Melonoma")
                    print("Melonoma Detected")
                elif cls2 == 2:
                    predict.append("Basal")
                    print("Basal Detected")
                if cls == 3:
                    predict.append("Acni")
                    pred="Acni"

                elif cls== 4:
                    predict.append("Pigmented")
                    pred="Pigmented Detected"

                elif cls == 5:
                    predict.append("Insect Bites")
                    pred="Insect Bites"

                else:
                    predict.append("Not Detected")
                    print("Not Detected")
                print(type(digit))
                act_digit = np.hstack((digit2[0][0:3],digit[0][3:]))

                data = {
                "ds": ["Actinic Keratosis", "Melonoma","Basal Cell", "Skin Acni", "Pigmented","Insect Bites"],
                "prob" : act_digit
                }
                df = pd.DataFrame(data)
                df.columns =["Diseases", "Predicted Percentage"]

                pred  = max(df['Predicted Percentage'])
                ind = df[df["Predicted Percentage"]==pred].index.values
                ind = ind.tolist()
                ind = ind[0]
                img_pred = df["Diseases"][ind]
                drug = treatment(img_pred)
                data = {
                "data":df, "img_name":img_pred, "drugs": drug
                }
                return data
            except Exception as e:
                    print("Error4: ",e)
        except Exception as e:
                    print("model error: ",e)
    except Exception as e:
        print("Error2: ",e)
