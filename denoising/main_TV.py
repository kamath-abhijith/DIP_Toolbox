## IMAGE DENOISING METHODS
#
#   Contents:
#       - Quadratic Smoothing
#       - Total Variation Denoising
#
#   Author: Abijith J Kamath
#   https://kamath-abhijth.github.io

# %%
import numpy as np
import scipy as sp

from skimage import io
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import rcParams
from patchify import patchify, unpatchify

# %% FUNCTION DEFINITION

# QUADRATIC SMOOTHING
def quad_smoothing(x, delta=1):
    ''' INPUT: Data vector, x
               Denoising weight, delta
    
        OUTPUT: Quadratic smoothing
                x^* = arg min || z - x ||^2 + delta || Dz ||^2,
                where D is the first difference operator
    '''
    
    N = len(x)
    D = sp.linalg.toeplitz(np.hstack(([-1],np.zeros(N-2))),np.hstack(([-1,1],np.zeros(N-2))))
    A = np.eye(N) + delta*np.matmul(D.T,D)

    return np.linalg.solve(A,x)

# TOTAL VARIATION DENOISING
def TV_denoising(x, delta=1):
    return


# %% MAIN
image = io.imread('images/lena.tif', 0)
m,n = image.shape

patch_size = (16,16)
patches = patchify(image, patch_size, step=1)

delta = 1
processed_patches = np.zeros(patches.shape)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        smooth_patch = quad_smoothing(patches[i][j].flatten(), delta)
        processed_patches[i][j] = smooth_patch.reshape(patch_size)

smooth_image = unpatchify(processed_patches, image.shape)

# %% PLOTS: QUADRATIC SMOOTHING
fig, plts = plt.subplots(1,2, figsize=(10,6))
plts[0].imshow(image, cmap='gray')
plts[1].imshow(smooth_image, cmap='gray')
plt.show()

# %%
