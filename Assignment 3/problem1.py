# # Spatial Domain Filtering
# 
# Contents:
# - Blurring using Mean Filtering
# - Sharpening using Unsharp Mask
# 
# Author: Abijith J. Kamath, IISc., https://kamath-abhijith.github.io

import numpy as np

from skimage import io
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams


# %%
## HELPER FUNCTIONS

# FULL-SCALE CONTRAST STRETCH
def fscs(image):
    image = np.double(image)

    return (255/(np.max(image)-np.min(image)))*(image - np.min(image))

# %%
# ## Mean Filtering
## FUNCTION DEFINITION: MEAN FILTERING

def mean_filter(image, kernel_size=5):
    blur_kernel = np.ones((kernel_size,kernel_size))/(kernel_size)**2

    return signal.convolve2d(image, blur_kernel, mode='same', boundary='fill', fillvalue=0)


# %%
## READ IMAGE
image = io.imread('noisy.tif', 0)
m,n = image.shape

## MEAN FILTERING
clean_image5 = mean_filter(image)
clean_image10 = mean_filter(image, 10)
clean_image15 = mean_filter(image, 15)


# %%
## FREQUENCY DOMAIN VIEW: MEAN FILTERING

fft_image = np.fft.fftshift(np.fft.fft2(image,(m,n)))

blur_kernel5 = np.ones((5,5))/(5)**2
blur_kernel10 = np.ones((10,10))/(10)**2
blur_kernel15 = np.ones((15,15))/(15)**2

fft_blur_kernel5 = np.fft.fftshift(np.fft.fft2(blur_kernel5,(m,n)))
fft_blur_kernel10 = np.fft.fftshift(np.fft.fft2(blur_kernel10,(m,n)))
fft_blur_kernel15 = np.fft.fftshift(np.fft.fft2(blur_kernel15,(m,n)))

fft_clean_image5 = np.fft.fftshift(np.fft.fft2(clean_image5,(m,n)))
fft_clean_image10 = np.fft.fftshift(np.fft.fft2(clean_image10,(m,n)))
fft_clean_image15 = np.fft.fftshift(np.fft.fft2(clean_image15,(m,n)))


# %%
## PLOTS: MEAN FILTERING
style.use('classic')

rcParams['text.usetex'] = True
rcParams.update({'font.size': 10})
rcParams['text.latex.preamble'] = [r'\usepackage{tgheros}'] 

fig, plts = plt.subplots(4,2,figsize=(10,15))
plts[0][0].imshow(image, vmin=0, vmax=255, cmap='gray')
plts[0][0].set_title(r"Noisy Image")
plts[0][0].set_xlabel(r"(a)")

plts[1][0].imshow(clean_image5, vmin=0, vmax=255, cmap='gray')
plts[1][0].set_title(r"Mean Filtered Image with $5\times 5$ blur kernel")
plts[1][0].set_xlabel(r"(b)")

plts[2][0].imshow(clean_image10, vmin=0, vmax=255, cmap='gray')
plts[2][0].set_title(r"Mean Filtered Image with $10\times 10$ blur kernel")
plts[2][0].set_xlabel(r"(c)")

plts[3][0].imshow(clean_image15, vmin=0, vmax=255, cmap='gray')
plts[3][0].set_title(r"Mean Filtered Image with $15\times 15$ blur kernel")
plts[3][0].set_xlabel(r"(d)")

plts[1][0].axis('off')

plts[0][1].imshow(np.log(1+np.abs(fft_image)), extent=[-m/2,m/2,-n/2,n/2])
plts[0][1].set_title(r"DFT of the image")
plts[0][1].set_xlabel(r"(e)")

plts[1][1].imshow(np.log(1+np.abs(fft_clean_image5)), extent=[-m/2,m/2,-n/2,n/2])
plts[1][1].set_title(r"DFT of image blurred with $5\times 5$ kernel")
plts[1][1].set_xlabel(r"(f)")

plts[2][1].imshow(np.log(1+np.abs(fft_clean_image10)), extent=[-m/2,m/2,-n/2,n/2])
plts[2][1].set_title(r"DFT of image blurred with $10\times 10$ kernel")
plts[2][1].set_xlabel(r"(g)")

plts[3][1].imshow(np.log(1+np.abs(fft_clean_image15)), extent=[-m/2,m/2,-n/2,n/2])
plts[3][1].set_title(r"DFT of image blurred with $15\times 15$ kernel")
plts[3][1].set_xlabel(r"(h)")

# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_3/Answers/figures/meanfilter.eps', format='eps')
plt.show()

# %%
## FUNCTION DEFINITION: UNSHARP MASKING

def unsharp_mask(image, blur_kernel_size=10, boost=1):

    blur_image = mean_filter(image, blur_kernel_size)
    sharp = image + boost*(image - blur_image)

    return fscs(sharp)


# %%
## UNSHARP MASKING

true_image = io.imread('characters.tif', 0)
blur_radius = 2

boost_vals = np.linspace(0,2)
mse_vals = []
for i in range(len(boost_vals)):
    sharp_image = unsharp_mask(clean_image10, blur_radius, boost_vals[i])
    mse_vals.append(np.square(sharp_image - true_image).mean())

lmse_idx = np.argmin(mse_vals)
sharp_image = unsharp_mask(clean_image10, blur_radius, boost_vals[lmse_idx])


# %%
## PLOTS: UNSHARP MASKING

fig, plts = plt.subplots(2,2,figsize=(10,10))
plts[0][0].imshow(clean_image10, vmin=0, vmax=255, cmap='gray')
plts[0][0].set_title(r"Blurred Image")
plts[0][0].set_xlabel(r"(a)")

plts[0][1].imshow(sharp_image, vmin=0, vmax=255, cmap='gray')
plts[0][1].set_title(r"Unsharp Masking with boost: $%.2f$"%(boost_vals[lmse_idx]))
plts[0][1].set_xlabel(r"(b)")

plts[1][0].plot(boost_vals, mse_vals, linewidth=2)
plts[1][0].grid()
plts[1][0].set_xlabel(r"Boost Scale")
plts[1][0].set_ylabel(r"Mean-Squared Error")
plts[1][0].set_xlabel(r"(c)")

plts[1][1].axis('off')

# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_3/Answers/figures/unsharp.eps', format='eps')
plt.show()