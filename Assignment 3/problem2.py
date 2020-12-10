# # Fourier-Domain Processing
# 
# Contents:
# - Ideal lowpass filtering
# - Gaussian filtering
# - Homoorphic processing
#
# Author: Abijith J. Kamath, IISc., https://kamath-abhijith.github.io

import numpy as np

from skimage import io
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

# %%
## CONSTRUCT IMAGE
M = N = 1001
m = np.arange(0,M)
n = np.arange(0,M)
mm, nn = np.meshgrid(m,n)

u0 = 100
v0 = 200
sine_image = np.sin(2*np.pi*u0*mm/(M) + 2*np.pi*v0*nn/(N))

## COMPUTE 2D DFT
fft_sine_image = np.fft.fft2(sine_image)
fft_sine_image = np.fft.fftshift(fft_sine_image)


# %%
## PLOTS
style.use('classic')

rcParams['text.usetex'] = True
rcParams.update({'font.size': 10})
rcParams['text.latex.preamble'] = [r'\usepackage{tgheros}'] 

fig, plts = plt.subplots(1,2,figsize=(20,12))
plts[0].imshow(np.abs(sine_image), cmap='gray')
plts[0].set_title(r"2D Sinusoid", fontsize=25)
plts[0].set_xlabel(r"(a)")
plts[1].imshow(np.log(1+np.abs(fft_sine_image)), cmap='gray', extent = [-M/2,M/2,-N/2,N/2])
plts[1].set_title(r"DFT of 2D Sinusoid", fontsize=25)
plts[1].set_xlabel(r"(b)")

# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_3/Answers/figures/visualisation.eps', format='eps')
plt.show()

# %%
## FUNCTION DEFINITON: IDEAL LOWPASS FILTER

def ideal_lowpass_filter(image, cutoff=10):
    m,n = image.shape

    ilpf_matrix = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if (i-m/2)**2 + (j-n/2)**2 <= cutoff**2:
                ilpf_matrix[i,j] = 1

    fft_image = np.fft.fft2(image)
    fft_image = np.fft.fftshift(fft_image)

    fft_lowpass_image = fft_image*ilpf_matrix
    lowpass_image = np.fft.ifft2(fft_lowpass_image)

    return np.abs(lowpass_image)


# %%
## IDEAL LOWPASS FILTER

image = io.imread('characters.tif', 0)

cutoff = 50
lowpass_image= ideal_lowpass_filter(image, cutoff)

fft_image = np.fft.fftshift(np.fft.fft2(image))
fft_lowpass_image = np.fft.fftshift(np.fft.fft2(lowpass_image))


# %%
## PLOTS: IDEAL LOWPASS FILTER

fig, plts = plt.subplots(2,2,figsize=(10,10))
plts[0][0].imshow(image, vmin=0, vmax=255, cmap='gray')
plts[0][0].set_title(r"Original Image")
plts[0][0].set_xlabel(r"(a)")

plts[0][1].imshow(lowpass_image, vmin=0, vmax=255, cmap='gray')
plts[0][1].set_title(r"Ideal Lowpass Image")
plts[0][1].set_xlabel(r"(b)")

plts[1][0].imshow(np.log(1+np.abs(fft_image)), extent = [-M/2,M/2,-N/2,N/2])
plts[1][0].set_title(r"DFT of the Image")
plts[1][0].set_xlabel(r"(c)")

plts[1][1].imshow(np.log(1+np.abs(fft_lowpass_image)), extent = [-M/2,M/2,-N/2,N/2])
plts[1][1].set_title(r"DFT of the Lowpass Image")
plts[1][1].set_xlabel(r"(d)")

# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_3/Answers/figures/lowpass.eps', format='eps')
plt.show()

# %%
## FUNCTION DEFINITION: GAUSSIAN FILTERING

def gaussian_filter(image, var=100):
    m,n = image.shape

    gaussian_matrix = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            d = (i-m/2)**2 + (j-n/2)**2
            gaussian_matrix[i,j] = np.exp(-d/(2*var**2))

    fft_image = np.fft.fft2(image)
    fft_image = np.fft.fftshift(fft_image)

    fft_smooth_image = fft_image*gaussian_matrix
    fft_smooth_image = np.fft.fftshift(fft_smooth_image)
    smooth_image = np.fft.ifft2(fft_smooth_image)

    return np.abs(smooth_image)


# %%
## GAUSSIAN FILTERING

image = io.imread('characters.tif', 0)

var = 10
smooth_image = gaussian_filter(image, var)

fft_image = np.fft.fftshift(np.fft.fft2(image))
fft_smooth_image = np.fft.fftshift(np.fft.fft2(smooth_image))


# %%
## PLOTS: GAUSSIAN FILTERING

fig, plts = plt.subplots(2,2,figsize=(10,10))
plts[0][0].imshow(image, vmin=0, vmax=255, cmap='gray')
plts[0][0].set_title(r"Original Image")
plts[0][0].set_xlabel(r"(a)")

plts[0][1].imshow(smooth_image, vmin=0, vmax=255, cmap='gray')
plts[0][1].set_title(r"Gaussian Blurred Image")
plts[0][1].set_xlabel(r"(b)")

plts[1][0].imshow(np.log(1+np.abs(fft_image)), extent = [-M/2,M/2,-N/2,N/2])
plts[1][0].set_title(r"DFT of the Image")
plts[1][0].set_xlabel(r"(c)")

plts[1][1].imshow(np.log(1+np.abs(fft_smooth_image)), extent = [-M/2,M/2,-N/2,N/2])
plts[1][1].set_title(r"DFT of the Smooth Image")
plts[1][1].set_xlabel(r"(d)")

# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_3/Answers/figures/gaussian.eps', format='eps')
plt.show()
