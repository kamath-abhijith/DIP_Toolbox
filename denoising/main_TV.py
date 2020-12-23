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

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

# %% FUNCTION DEFINITION

# CLIPPING FUNCTION
def clip(b, T):
    out = np.zeros(len(b))
    for i in range(len(b)):
        if np.abs(b[i]) <= T:
            out[i] = b[i]
        else:
            out[i] = T*np.sign(b[i])
    
    return out

# QUADRATIC SMOOTHING
def quad_smoothing(x, delta=1):
    ''' INPUT: Measurement, x
               Denoising weight, delta
    
        OUTPUT: Quadratic smoothing
                z^* = arg min || z - x ||^2 + delta || Dz ||^2,
                where D is the first difference operator
    '''
    
    N = len(x)
    D = sp.linalg.toeplitz(np.hstack(([-1],np.zeros(N-2))),np.hstack(([-1,1],np.zeros(N-2))))
    A = np.eye(N) + delta*np.matmul(D.T,D)

    return np.linalg.solve(A,x)

# TOTAL VARIATION DENOISING
def clipping_TV(y, lambd=1, alpha=1, tol=1e-6, max_iterations=25):
    ''' INPUT: Measurement, y
               Denoising weight, lambda
               Majoriser weight, alpha
    
        OUTPUT: Total Variation Denoising
                z^* = arg min || z - x ||^2 + lambd || Dz ||^1,
                where D is the first difference operator
    '''

    N = len(y)
    D = sp.linalg.toeplitz(np.hstack(([-1],np.zeros(N-2))),np.hstack(([-1,1],np.zeros(N-2))))

    z = np.zeros(N-1)
    for _ in range(max_iterations):
        zcopy = z

        x = y - (D.T).dot(z)
        z = clip(z+D.dot(x)/alpha, lambd/2)
        if np.linalg.norm(z-zcopy)<=tol:
            break

    return x


# %% MAIN
image = io.imread('images/house.tif', 0)
m,n = image.shape

sigma = 10
noisy_image = image + sigma*np.random.randn(m,n)

patch_size = (8,8)
patches = extract_patches_2d(noisy_image, patch_size)

lambd = 10
alpha = 2
processed_patches = np.zeros(patches.shape)
for i in range(patches.shape[0]):
    # smooth_patch = quad_smoothing(patches[i].flatten(), delta)
    smooth_patch = clipping_TV(patches[i].flatten(), lambd, alpha)
    processed_patches[i] = smooth_patch.reshape(patch_size)

smooth_image = reconstruct_from_patches_2d(processed_patches, image.shape)

# %% PLOTS: QUADRATIC SMOOTHING
fig, plts = plt.subplots(1,3, figsize=(15,6))
plts[0].imshow(image, cmap='gray')
plts[1].imshow(noisy_image, cmap='gray')
plts[2].imshow(smooth_image, cmap='gray')
plt.show()

# %%
