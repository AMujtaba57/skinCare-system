######## Import Libraries #########

import cv2
import sys
import imageio
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from .function_file import *
######## Background Removal #######

class skinDetector(object):

        #class constructor
        def __init__(self, imageName):
            self.image = cv2.imread(imageName)
            if self.image is None:
                print("IMAGE NOT FOUND")
                exit(1)
            #self.image = cv2.resize(self.image,(600,600),cv2.INTER_AREA)
            self.HSV_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            self.YCbCr_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
            self.binary_mask_image = self.HSV_image
        #================================================================================================================================
        #function to process the image and segment the skin using the HSV and YCbCr colorspaces, followed by the Watershed algorithm
        def find_skin(self):
            self.__color_segmentation()
            output=self.__region_based_segmentation()
            return output

        #================================================================================================================================
        #Apply a threshold to an HSV and YCbCr images, the used values were based on current research papers along with some
        # empirical tests and visual evaluation
        def __color_segmentation(self):
            lower_HSV_values = np.array([0, 40, 0], dtype = "uint8")
            upper_HSV_values = np.array([25, 255, 255], dtype = "uint8")

            lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
            upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

            #A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
            mask_YCbCr = cv2.inRange(self.YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
            mask_HSV = cv2.inRange(self.HSV_image, lower_HSV_values, upper_HSV_values)

            self.binary_mask_image = cv2.add(mask_HSV,mask_YCbCr)

        #================================================================================================================================
        #Function that applies Watershed and morphological operations on the thresholded image
        def __region_based_segmentation(self):
            #morphological operations
            image_foreground = cv2.erode(self.binary_mask_image,None,iterations = 3)     	#remove noise
            dilated_binary_image = cv2.dilate(self.binary_mask_image,None,iterations = 3)   #The background region is reduced a little because of the dilate operation
            ret,image_background = cv2.threshold(dilated_binary_image,1,128,cv2.THRESH_BINARY)  #set all background regions to 128

            image_marker = cv2.add(image_foreground,image_background)   #add both foreground and backgroud, forming markers. The markers are "seeds" of the future image regions.
            image_marker32 = np.int32(image_marker) #convert to 32SC1 format

            cv2.watershed(self.image,image_marker32)
            m = cv2.convertScaleAbs(image_marker32) #convert back to uint8

            #bitwise of the mask with the input image
            ret,image_mask = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            output = cv2.bitwise_and(self.image,self.image,mask = image_mask)

            #show the images
            self.show_image(self.image)
            self.show_image(image_mask)
            self.show_image(output)
            cv2.imwrite("mask3.jpg", output)
            return output

        def show_image(self, image):
            plt.rcParams['image.interpolation'] = 'nearest'
            plt.rcParams['figure.dpi'] = 200


            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

def background_removal(filename):

    detector = skinDetector(filename)
    output=detector.find_skin()
    # cv2.imwrite("output", output)
    return output

########## Facial Detection  ##########
def facial_features_extraction(filename):
        # Load the jpg file into a numpy array
#         image = face_recognition.load_image_file(filename)

        # Find all facial features in all the faces in the image
        face_landmarks_list = face_recognition.face_landmarks(filename)

        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
        if len(face_landmarks_list)==0:
            img = hair_removal(filename)
            return img
        else:
            # Create a PIL imagedraw object so we can draw on the picture
            pil_image = Image.fromarray(filename)
            d = ImageDraw.Draw(pil_image)

            for face_landmarks in face_landmarks_list:

                # Print the location of each facial feature in this image
                for facial_feature in face_landmarks.keys():
                    print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

                # Let's trace out each facial feature in the image with a line!
                for facial_feature in face_landmarks.keys():
                    if facial_feature=='left_eye':
                        x=[]
                        x.append(face_landmarks[facial_feature][0][0])
                        k=int(len(face_landmarks[facial_feature])/2)
                        x.append(face_landmarks[facial_feature][k-1][1]-3)
                        x.append(face_landmarks[facial_feature][k][0])
                        x.append(face_landmarks[facial_feature][-1][1]+3)

            #             xy=[(232,  158-3), (264, 168+3)]
                        d.polygon(face_landmarks[facial_feature],fill='black',outline='black')
                        d.rectangle(x,fill='black')
                    elif facial_feature=='right_eye':
                        x=[]
                        x.append(face_landmarks[facial_feature][0][0])
                        k=int(len(face_landmarks[facial_feature])/2)
                        x.append(face_landmarks[facial_feature][k-1][1]-3)
                        k=int(len(face_landmarks[facial_feature])/2)
                        x.append(face_landmarks[facial_feature][k][0])
                        x.append(face_landmarks[facial_feature][-1][1]+3)

                        d.rectangle(x,fill='black')
                        d.polygon(face_landmarks[facial_feature],fill='black',outline='black')
                    elif facial_feature=='left_eyebrow':
                        d.polygon(face_landmarks[facial_feature],fill='black',outline='black')
                    elif facial_feature=='right_eyebrow':
                        d.polygon(face_landmarks[facial_feature],fill='black',outline='black')
                    elif facial_feature=='top_lip':
                        x=[]
                        x.append(face_landmarks[facial_feature][-1][0])
                        d.polygon(face_landmarks[facial_feature],fill='black',outline='black')
                    elif facial_feature=='bottom_lip':
                        x.append(face_landmarks[facial_feature][-1][1])
                        x.append(face_landmarks[facial_feature][-1][0])
                        k=int(len(face_landmarks[facial_feature])/2)
                        x.append(face_landmarks[facial_feature][k-1][1])

                        d.polygon(face_landmarks[facial_feature],fill='black',outline='black')
                        d.rectangle(x,fill='black')

            #         else:
            #             d.line(face_landmarks[facial_feature], width=5)
            return pil_image

def face_facial(filename):
    pil_image=facial_features_extraction(filename)
    imageio.imwrite("facial3.jpg",pil_image)
    return pil_image

####### Main part ######
