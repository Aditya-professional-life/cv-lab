---------------------------------------    6       ---------------------------------------------------------------------
 

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
import cv2


img = plt.imread("earcell.webp")
img.astype(float)
plt.imshow(img)
plt.title(" Orginal Image")


f1 = np.fft.fft2(img)
plt.imshow(np.abs(f1))
plt.title("Frequency Spectrum ")


f2 = np.fft.fftshift(f1)
plt.imshow(np.abs(f2))
plt.title("Centerd  Spectrum")


f3 = np.log(1+np.abs(f2))
plt.imshow(f3)
plt.title("1+abs(f2)")


l_fft = np.fft.fft2(f1)
l1 = np.real(l_fft)
plt.imshow(l1)
plt.title("2-D")








------------------------------------------         7 -------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import cv2 as cv

image = plt.imread("earcell.webp")
plt.imshow(image)
plt.title("Orginal Image")

gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
plt.imshow(gray_image,cmap = "gray")

gray_image.shape

gray_cropped = gray_image[50:1300,50:1500]
plt.imshow(gray_cropped,cmap = "gray")

mean_image = np.mean(gray_cropped)
print(f"The Mean Coefflicent is {mean_image}")

std_image = np.std(gray_cropped)
print(f"The std Coefflicent is {std_image}")

pearson,_ = pearsonr(gray_cropped.flatten(),gray_image[50:1300,50:1500].flatten())
print(f"The pearson Coefflicent is {pearson}")



------------------------------------------- 8 --------------------------------------------------




import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, median_filter
from skimage.util import random_noise

img = plt.imread("neuron.jpg")
plt.imshow(img)
plt.title("Orginal Image")



gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
plt.imshow(gray_img,cmap = "gray")
plt.title("GrayScale")



salt_pepper_image = random_noise(gray_img,mode="s&p",amount =0.05)
salt_pepper_image = np.array(255*salt_pepper_image,dtype = 'uint8')
plt.imshow(salt_pepper_image,cmap="gray")
plt.title("Salt and Pepper Noise")

f = median_filter(salt_pepper_image,size = 3)
plt.imshow(f,cmap ="gray")
plt.title("3X# filter")




f1 = median_filter(salt_pepper_image,size = 10)
plt.imshow(f1,cmap = "gray")
plt.title("10X10 filter")

image = plt.imread("earcell.webp")
plt.imshow(image)
image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
plt.title("Original Image")


a = np.array([[0.001,0.001,0.001],
              [0.001,0.001,0.001],
              [0.001,0.001,0.001]])
R1 = convolve(image , a)
plt.imshow(R1,cmap ="gray")
plt.title("First CNN layer")


a = np.array([[0.005,0.005,0.005],
              [0.005,0.005,0.005],
              [0.005,0.005,0.005]])
R2 = convolve(image , a)
plt.imshow(R2,cmap ="gray")
plt.title("2nd CNN layer")


----------------------------------------- 9    -------------------------------------------

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


image = plt.imread("neuron.jpg")
plt.imshow(image)
plt.title("Original Image")

gray_image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
plt.imshow(gray_image)
plt.title("Gray Image")




laplacian_image = cv.Laplacian(gray_image, cv.CV_64F)
plt.imshow(laplacian_image,cmap="gray")
plt.title("laplacian_image")


x = np.array([[1,0,-1],
             [1,0,-1],
              [1,0,-1]])
y = np.array([[1,1,1],
             [0,0,0],
              [-1,-1,-1]])

prewitt_x = cv.filter2D(gray_image,-1,x)
prewitt_y = cv.filter2D(gray_image,-1,y)
prewitt_img = np.sqrt(prewitt_x^2 +prewitt_y^2)
plt.imshow(prewitt_img)
plt.title("prewitt_img")


x = np.array([[1,0],
             [0,-1]])
y = np.array([[0,1],
             [-1,0]])

robert_x = cv.filter2D(gray_image,-1,x)
robert_y = cv.filter2D(gray_image,-1,y)
robert_image = np.sqrt(robert_x^2+robert_y^2)
plt.imshow(robert_image)
plt.title("robert_image")


sobel_x = cv.Sobel(gray_image,cv.CV_64F,1,0,ksize = 3)
sobel_y = cv.Sobel(gray_image,cv.CV_64F,0,1,ksize = 3)


sobel_horizontal = np.abs(sobel_x)
plt.imshow(sobel_horizontal)
plt.title("Sobel Horizontal Image")


sobel_vertical = np.abs(sobel_y)
plt.imshow(sobel_vertical)
plt.title("Sobel Vertical Image")

sobel_image = np.sqrt(prewitt_x^2+prewitt_y^2)
plt.imshow(sobel_image)
plt.title("Sobel Image")
