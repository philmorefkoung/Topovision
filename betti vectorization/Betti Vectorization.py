# Libraries
import cv2 # opencv-python==4.10.0.84
import numpy as np
import pandas as pd
from gtda.homology import CubicalPersistence # giotto-tda==0.6.2
from gtda.diagrams import BettiCurve

#  Function to compute Betti vectors
#  'images' should be a numpy array
def generate_betti_vectors(images):
    CP0 = CubicalPersistence(homology_dimensions=[0], coeff=3, n_jobs=-1)
    CP1 = CubicalPersistence(homology_dimensions=[1], coeff=3, n_jobs=-1)
    BC = BettiCurve(n_bins=50)

    def generate_red0(img):
        red_channel = img[:, :, 2]
        diagram_h1_1 = CP0.fit_transform(np.array(red_channel)[None, :, :])
        y_betti_curves_red = BC.fit_transform(diagram_h1_1)
        return np.reshape(y_betti_curves_red, 50).tolist()

    betti0_red = [generate_red0(img) for img in images]
    
    def generate_red1(img):
        red_channel = img[:, :, 2]
        diagram_h1_1 = CP1.fit_transform(np.array(red_channel)[None, :, :])
        y_betti_curves_red = BC.fit_transform(diagram_h1_1)
        return np.reshape(y_betti_curves_red, 50).tolist()
        
    betti1_red = [generate_red1(img) for img in images]

    def generate_green0(img):
        green_channel = img[:, :, 1]
        diagram_h1_2 = CP0.fit_transform(np.array(green_channel)[None, :, :])
        y_betti_curves_green = BC.fit_transform(diagram_h1_2)
        return np.reshape(y_betti_curves_green, 50).tolist()
 
    betti0_green = [generate_green0(img) for img in images]

    def generate_green1(img):
        green_channel = img[:, :, 1]
        diagram_h1_2 = CP1.fit_transform(np.array(green_channel)[None, :, :])
        y_betti_curves_green = BC.fit_transform(diagram_h1_2)
        return np.reshape(y_betti_curves_green, 50).tolist()
 
    betti1_green = [generate_green1(img) for img in images]

    def generate_blue0(img):
        blue_channel = img[:, :, 0]
        diagram_h1_3 = CP0.fit_transform(np.array(blue_channel)[None, :, :])
        y_betti_curves_blue = BC.fit_transform(diagram_h1_3)
        return np.reshape(y_betti_curves_blue, 50).tolist()
 
    betti0_blue = [generate_blue0(img) for img in images]

    def generate_blue1(img):
        blue_channel = img[:, :, 0]
        diagram_h1_3 = CP1.fit_transform(np.array(blue_channel)[None, :, :])
        y_betti_curves_blue = BC.fit_transform(diagram_h1_3)
        return np.reshape(y_betti_curves_blue, 50).tolist()
 
    betti1_blue = [generate_blue1(img) for img in images]

    def generate_gray0(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Convert to grayscale
        diagram_h1_0 = CP0.fit_transform(np.array(img_gray)[None, :, :])
        y_betti_curves_gray = BC.fit_transform(diagram_h1_0)
        return np.reshape(y_betti_curves_gray, 50).tolist()
 
    betti0_gray = [generate_gray0(img) for img in images]

    def generate_gray1(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Convert to grayscale
        diagram_h1_0 = CP1.fit_transform(np.array(img_gray)[None, :, :]) 
        y_betti_curves_gray = BC.fit_transform(diagram_h1_0)
        return np.reshape(y_betti_curves_gray, 50).tolist()
 
    betti1_gray = [generate_gray1(img) for img in images]

    # Convert to DataFrame
    betti0_gray = pd.DataFrame(betti0_gray)
    betti1_gray = pd.DataFrame(betti1_gray)
    betti0_red = pd.DataFrame(betti0_red)
    betti1_red = pd.DataFrame(betti1_red)
    betti0_blue = pd.DataFrame(betti0_blue)
    betti1_blue = pd.DataFrame(betti1_blue)
    betti0_green = pd.DataFrame(betti0_green)
    betti1_green = pd.DataFrame(betti1_green)
    
    combined = pd.concat([betti0_gray,
                          betti1_gray,
                          betti0_red, 
                          betti1_red,
                          betti0_blue,
                          betti1_blue,
                          betti0_green, 
                          betti1_green], 
                          axis=1)
    return combined

betti_vectors = generate_betti_vectors(images)
betti_vectors.to_csv('betti_vectors', index=False)
