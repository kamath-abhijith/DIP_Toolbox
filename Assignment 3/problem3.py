# ## Homomorphic Processing
# 
# Contents:
# - Homomorphic processing
#
# Author: Abijith J. Kamath, IISc., https://kamath-abhijith.github.io

import numpy as np

from skimage import io
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams

# %%
## FUNCTION DEFINITION: HOMOMOPRHIC PROCESSING

def homomorphic_gaussian_filter(image, gamma=(1,2), var=300):
    m,n = image.shape
    
    homo_filter = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            d = (i-m/2.0)**2 + (j-n/2.0)**2
            homo_filter[i,j] = (gamma[1]-gamma[0])*(1-np.exp(-d/(2*var**2))) + gamma[0]

    log_image = np.log1p(np.float64(image))
    fft_log_image = np.fft.fft2(log_image)
    fft_log_image = np.fft.fftshift(fft_log_image)

    fft_homo_image = fft_log_image*homo_filter
    fft_homo_image = np.fft.ifftshift(fft_homo_image)
    homo_image = np.fft.ifft2(fft_homo_image)

    return np.exp(np.real(homo_image), dtype=np.float64)-1


# %%
## HOMOMORPHIC PROCESSING

image = io.imread('PET_image.tif', 0)

gamma = (0.5,1)
var = 10
contrast_enhanced_image = homomorphic_gaussian_filter(image, gamma, var)


# %%
## PLOTS: HOMOMORPHIC PROCESSING
style.use('classic')

rcParams['text.usetex'] = True
rcParams.update({'font.size': 10})
rcParams['text.latex.preamble'] = [r'\usepackage{tgheros}'] 

fig, plts = plt.subplots(1,2,figsize=(10,6))
plts[0].imshow(image, vmin=0, vmax=255, cmap='gray')
plts[0].set_title(r"Original Image")
plts[0].set_xlabel(r"(a)")
plts[1].imshow(contrast_enhanced_image, cmap='gray')
plts[1].set_title(r"Homomorphic Contrast Enhanced Image")
plts[1].set_xlabel(r"(b)")

# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_3/Answers/figures/homomorphic.eps', format='eps')
plt.show()