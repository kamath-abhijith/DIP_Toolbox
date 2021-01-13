# # Image Decimation
#
# Author: Abijith J. Kamath
# https://kamath-abhijith.github.io

# %%
import numpy as np
import cv2

from skimage import io
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import style

# ## Decimation and Filtering

## FUNCTION DEFINITON :: MEDIAN AND GAUSSIAN FILTERING

def gaussian_filter(image, kernel_size, var):
    m = kernel_size[0]
    n = kernel_size[1]
    gaussian_matrix = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            d = (i-m/2)**2 + (j-n/2)**2
            gaussian_matrix[i,j] = np.exp(-d/(2*var**2))

    return signal.convolve2d(image, gaussian_matrix, mode='same', boundary='fill', fillvalue=0)

# %%
## FUNCTION DEFINITION :: DECIMATION

def decimate_image(image, decimation_factor):
    m,n = image.shape
    return image[0:m:decimation_factor, 0:n:decimation_factor]


# %%
## MAIN :: DECIMATION

image = io.imread('barbara.tif', 0)

decimation_factor = 2
decimated_image = decimate_image(image, decimation_factor)

var = 2
kernel_size = (3,3)
decimated_image_postp = decimate_image(gaussian_filter(image, kernel_size, var), decimation_factor)

resized_image = cv2.resize(image, dsize=(int(image.shape[1]/decimation_factor), int(image.shape[0]/decimation_factor)), interpolation=cv2.INTER_CUBIC)


# %%
## PLOTS :: DECIMATION

fig, plts = plt.subplots(2,2,figsize=(10,12))
plts[0][0].imshow(image, cmap='gray')
# plts[0][0].set_xlim([0,image.shape[1]])
# plts[0][0].set_ylim([image.shape[0],0])
plts[0][0].set_title(r"Original Image")

plts[0][1].imshow(decimated_image, cmap='gray')
# plts[0][1].set_xlim([0,image.shape[1]])
# plts[0][1].set_ylim([image.shape[0],0])
plts[0][1].set_title(r"Decimated Image")

plts[1][0].imshow(decimated_image_postp, cmap='gray')
# plts[1][0].set_xlim([0,image.shape[1]])
# plts[1][0].set_ylim([image.shape[0],0])
plts[1][0].set_title(r"Decimated Image After Gaussian Blur")

plts[1][1].imshow(resized_image, cmap='gray')
# plts[1][1].set_xlim([0,image.shape[1]])
# plts[1][1].set_ylim([image.shape[0],0])
plts[1][1].set_title(r"Decimated Image Using resize")

# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_4/Answers/figures/decimation.eps', format='eps')
plt.show()