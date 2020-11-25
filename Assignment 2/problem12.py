# # Image Contrast Enhancement
# 
# Contents:
# - Full-Scale Contrast Stretch
# - Gamma Transforms
# - Histogram Equalisation
# - Adaptive Histogram Equalisation (CLAHE)
# 
# Author: Abijith J. Kamath, IISc.
# https://kamath-abhijith.github.io

# %%
import numpy as np

from skimage import io
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import style

# %%
## FUNCTION DEFINITION: FULL-SCALE CONTRAST STRETCH
def fscs(image):
    image = np.double(image)

    return (255/(np.max(image)-np.min(image)))*(image - np.min(image))

## FUNCTION DEFINITION: GAMMA TRANSFORM
def gamma_transform(image, gamma=0.5):
    return ((np.double(image)/255.0)**gamma)*255.0

## FUNCTION DEFINITION: HISTOGRAM EQUALISATION
def hist_eq(image):
    [m,n] = image.shape

    [image_hist, _] = np.histogram(image, bins=np.arange(0,255,1))
    image_density = image_hist/(m*n)

    sum_image_density = np.cumsum(image_density)
    sum_image_density = np.append(sum_image_density, [0,0])
    hist_eq_image = np.zeros((m,n), dtype=np.int)
    for i in range(m):
        for j in range(n):
            hist_eq_image[i,j] = 255.0*sum_image_density[image[i,j]]

    return hist_eq_image

## FUNCTION DEFINITION: CLAHE
def interpolate(subBin,LU,RU,LB,RB,subX,subY):
    subImage = np.zeros(subBin.shape)
    num = subX*subY
    for i in range(subX):
        inverseI = subX-i
        for j in range(subY):
            inverseJ = subY-j
            val = subBin[i,j].astype(int)
            subImage[i,j] = np.floor((inverseI*(inverseJ*LU[val] + j*RU[val])+ i*(inverseJ*LB[val] + j*RB[val]))/float(num))
    return subImage

def clahe(img,clipLimit,nrBins=128,nrX=0,nrY=0):
    
    h,w = img.shape
    if clipLimit==1:
        return
    nrBins = max(nrBins,128)
    if nrX==0:
        xsz = 32
        ysz = 32
        nrX = np.ceil(h/xsz)#240
        #Excess number of pixels to get an integer value of nrX and nrY
        excX= int(xsz*(nrX-h/xsz))
        nrY = np.ceil(w/ysz)#320
        excY= int(ysz*(nrY-w/ysz))
        #Pad that number of pixels to the image
        if excX!=0:
            img = np.append(img,np.zeros((excX,img.shape[1])).astype(int),axis=0)
        if excY!=0:
            img = np.append(img,np.zeros((img.shape[0],excY)).astype(int),axis=1)
    else:
        xsz = round(h/nrX)
        ysz = round(w/nrY)
    
    nrPixels = xsz*ysz
    xsz2 = round(xsz/2)
    ysz2 = round(ysz/2)
    claheimg = np.zeros(img.shape)
    
    if clipLimit > 0:
        clipLimit = max(1,clipLimit*xsz*ysz/nrBins)
    else:
        clipLimit = 50
    
    minVal = 0
    maxVal = 255
    
    binSz = np.floor(1+(maxVal-minVal)/float(nrBins))
    LUT = np.floor((np.arange(minVal,maxVal+1)-minVal)/float(binSz))
    
    #BACK TO CLAHE
    bins = LUT[img]
    # print(bins.shape)
    hist = np.zeros((nrX,nrY,nrBins))
    print(nrX,nrY,hist.shape)
    for i in range(nrX):
        for j in range(nrY):
            bin_ = bins[i*xsz:(i+1)*xsz,j*ysz:(j+1)*ysz].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i,j,bin_[i1,j1]]+=1
    
    if clipLimit>0:
        for i in range(nrX):
            for j in range(nrY):
                nrExcess = 0
                for nr in range(nrBins):
                    excess = hist[i,j,nr] - clipLimit
                    if excess>0:
                        nrExcess += excess
                
                binIncr = nrExcess/nrBins
                upper = clipLimit - binIncr
                for nr in range(nrBins):
                    if hist[i,j,nr] > clipLimit:
                        hist[i,j,nr] = clipLimit
                    else:
                        if hist[i,j,nr]>upper:
                            nrExcess += upper - hist[i,j,nr]
                            hist[i,j,nr] = clipLimit
                        else:
                            nrExcess -= binIncr
                            hist[i,j,nr] += binIncr
                
                if nrExcess > 0:
                    stepSz = max(1,np.floor(1+nrExcess/nrBins))
                    for nr in range(nrBins):
                        nrExcess -= stepSz
                        hist[i,j,nr] += stepSz
                        if nrExcess < 1:
                            break
    
    map_ = np.zeros((nrX,nrY,nrBins))
    #print(map_.shape)
    scale = (maxVal - minVal)/float(nrPixels)
    for i in range(nrX):
        for j in range(nrY):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i,j,nr]
                map_[i,j,nr] = np.floor(min(minVal+sum_*scale,maxVal))
    
    xI = 0
    for i in range(nrX+1):
        if i==0:
            subX = int(xsz/2)
            xU = 0
            xB = 0
        elif i==nrX:
            subX = int(xsz/2)
            xU = nrX-1
            xB = nrX-1
        else:
            subX = xsz
            xU = i-1
            xB = i
        
        yI = 0
        for j in range(nrY+1):
            if j==0:
                subY = int(ysz/2)
                yL = 0
                yR = 0
            elif j==nrY:
                subY = int(ysz/2)
                yL = nrY-1
                yR = nrY-1
            else:
                subY = ysz
                yL = j-1
                yR = j
            UL = map_[xU,yL,:]
            UR = map_[xU,yR,:]
            BL = map_[xB,yL,:]
            BR = map_[xB,yR,:]
            subBin = bins[xI:xI+subX,yI:yI+subY]
            subImage = interpolate(subBin,UL,UR,BL,BR,subX,subY)
            claheimg[xI:xI+subX,yI:yI+subY] = subImage
            yI += subY
        xI += subX
        
    return claheimg

# %%
## FUNCTION DEFINITION: SATURATED CONTRAST STRETCHING

def saturated_contrast_stretching(image, thresholds=(10, 90)):
    
    llim = thresholds[0]*255.0/100
    ulim = thresholds[1]*255.0/100

    zero_image = (image<=llim)*1.0
    ones_image = (image<=ulim)*1.0

    set_zero_image = image*np.logical_not(zero_image)
    set_ones_image = set_zero_image*np.logical_not(ones_image) + np.logical_and(set_zero_image, ones_image)*255.0
    
    return fscs(set_ones_image)


# %%
## READ/DISPLAY IMAGES WITH THEIR HISTOGRAMS

image1 = io.imread("LowLight_1.png", 0)
image2 = io.imread("LowLight_2.png", 0)
image3 = io.imread("LowLight_3.png", 0)
image4 = io.imread("Hazy.png", 0)
image5 = io.imread("StoneFace.png", 0)

numbins = 255
[image1_hist, bins] = np.histogram(image1, bins=np.arange(0,numbins,1))
[image2_hist, bins] = np.histogram(image2, bins=np.arange(0,numbins,1))
[image3_hist, bins] = np.histogram(image3, bins=np.arange(0,numbins,1))
[image4_hist, bins] = np.histogram(image4, bins=np.arange(0,numbins,1))
[image5_hist, bins] = np.histogram(image5, bins=np.arange(0,numbins,1))


# %%
## PLOTS
style.use('classic')

rcParams['text.usetex'] = True
rcParams.update({'font.size': 10})
rcParams['text.latex.preamble'] = [r'\usepackage{tgheros}'] 


# %%
## FULL-SCALE CONTRAST STRETCH

image1_fscs = fscs(image1)
image2_fscs = fscs(image2)

[image1_fscs_hist, bins] = np.histogram(image1_fscs, bins=np.arange(0,numbins,1))
[image2_fscs_hist, bins] = np.histogram(image2_fscs, bins=np.arange(0,numbins,1))

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image1, cmap='gray')
plts[0].set_ylabel(r"Low Light 1")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image1_fscs, cmap='gray')
plts[1].set_xlabel(r"Linear Constrast Stretching")

plts[2].plot(image1_fscs_hist, '-', color='green', label=r'Linear Contrast Stretching')
plts[2].plot(image1_hist, '-', color='blue', label=r'Original')
plts[2].set_xlim([0, 255])
plts[2].legend()
plts[2].grid('on')
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/lowlight1_fscs.eps', format='eps')


ffig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image2, cmap='gray')
plts[0].set_ylabel(r"Low Light 2")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image2_fscs, cmap='gray')
plts[1].set_xlabel(r"Linear Constrast Stretching")

plts[2].plot(image2_fscs_hist, '-', color='green', label=r'Linear Contrast Stretching')
plts[2].plot(image2_hist, '-', color='blue', label=r'Original')
plts[2].set_xlim([0, 255])
plts[2].legend()
plts[2].grid('on')
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/lowlight2_fscs.eps', format='eps')

plt.show()


# %%
## GAMMA TRANSFORMATION

image1_gamma = gamma_transform(image1)
image2_gamma = gamma_transform(image2)
image4_gamma = gamma_transform(image4)

[image1_gamma_hist, bins] = np.histogram(image1_gamma, bins=np.arange(0,numbins,1))
[image2_gamma_hist, bins] = np.histogram(image2_gamma, bins=np.arange(0,numbins,1))
[image4_gamma_hist, bins] = np.histogram(image4_gamma, bins=np.arange(0,numbins,1))

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image1, cmap='gray')
plts[0].set_ylabel(r"Low Light 1")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image1_gamma, cmap='gray')
plts[1].set_xlabel(r"Gamma Transform")

plts[2].plot(image1_gamma_hist, '-', color='green', label=r'Gamma Transform')
plts[2].plot(image1_hist, '-', color='blue', label=r'Original')
plts[2].set_xlim([0, 255])
plts[2].legend()
plts[2].grid('on')
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/lowlight1_gamma.eps', format='eps')

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image2, cmap='gray')
plts[0].set_ylabel(r"Low Light 2")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image2_gamma, cmap='gray')
plts[1].set_xlabel(r"Gamma Transform")

plts[2].plot(image2_gamma_hist, '-', color='green', label=r'Gamma Transform')
plts[2].plot(image2_hist, '-', color='blue', label=r'Original')
plts[2].set_xlim([0, 255])
plts[2].legend()
plts[2].grid('on')
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/lowlight2_gamma.eps', format='eps')

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image4, cmap='gray')
plts[0].set_ylabel(r"Hazy")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image4_gamma, cmap='gray')
plts[1].set_xlabel(r"Gamma Transform")

plts[2].plot(image4_gamma_hist, '-', color='green', label=r'Gamma Transform')
plts[2].plot(image4_hist, '-', color='blue', label=r'Original')
plts[2].set_xlim([0, 255])
plts[2].legend()
plts[2].grid('on')
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/hazy_gamma.eps', format='eps')

plt.show()


# %%
## HISTOGRAM EQUALISATION

image2_histeq = hist_eq(image2)
image3_histeq = hist_eq(image3)
image4_histeq = hist_eq(image4)
image5_histeq = hist_eq(image5)

[image2_histeq_hist, bins] = np.histogram(image2_histeq, bins=np.arange(0,numbins,1))
[image3_histeq_hist, bins] = np.histogram(image3_histeq, bins=np.arange(0,numbins,1))
[image4_histeq_hist, bins] = np.histogram(image4_histeq, bins=np.arange(0,numbins,1))
[image5_histeq_hist, bins] = np.histogram(image5_histeq, bins=np.arange(0,numbins,1))

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image2, cmap='gray')
plts[0].set_ylabel(r"Low Light 2")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image2_histeq, cmap='gray')
plts[1].set_xlabel(r"Histogram Equalisation")

plts[2].plot(image2_histeq_hist, '-', color='green', label=r'Histogram Equalisation')
plts[2].plot(image2_hist, '-', color='blue', label=r'Original')
plts[2].set_xlim([0, 255])
plts[2].legend()
plts[2].grid('on')
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/lowlight2_histeq.eps', format='eps')

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image3, cmap='gray')
plts[0].set_ylabel(r"Low Light 3")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image3_histeq, cmap='gray')
plts[1].set_xlabel(r"Histogram Equalisation")

plts[2].plot(image3_histeq_hist, '-', color='green', label=r'Histogram Equalisation')
plts[2].plot(image3_hist, '-', color='blue', label=r'Original')
plts[2].set_xlim([0, 255])
plts[2].legend()
plts[2].grid('on')
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/lowlight3_histeq.eps', format='eps')

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image4, cmap='gray')
plts[0].set_ylabel(r"Hazy")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image4_histeq, cmap='gray')
plts[1].set_xlabel(r"Histogram Equalisation")

plts[2].plot(image4_histeq_hist, '-', color='green', label=r'Histogram Equalisation')
plts[2].plot(image4_hist, '-', color='blue', label=r'Original')
plts[2].set_xlim([0, 255])
plts[2].legend()
plts[2].grid('on')
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/hazy_histeq.eps', format='eps')

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image5, cmap='gray')
plts[0].set_ylabel(r"Stone Face")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image5_histeq, cmap='gray')
plts[1].set_xlabel(r"Histogram Equalisation")

plts[2].plot(image5_histeq_hist, '-', color='green', label=r'Histogram Equalisation')
plts[2].plot(image5_hist, '-', color='blue', label=r'Original')
plts[2].set_xlim([0, 255])
plts[2].legend()
plts[2].grid('on')
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/stoneface_histeq.eps', format='eps')

plt.show()


# %%
## CLAHE

image5_clahe = clahe(image5, 0.9, 255, 8, 8)

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image5, cmap='gray')
plts[0].set_ylabel(r"Stone Face")
plts[0].set_xlabel(r"Original Image")

plts[1].imshow(image5_clahe, cmap='gray')
plts[1].set_xlabel(r"CLAHE")

plts[2].imshow(image5_histeq, cmap='gray')
plts[2].set_xlabel(r"Histogram Equalisation")
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/stoneface_clahe.eps', format='eps')

plt.show()


# %%
## SATURATED CONTRAST STRETCHING

image = io.imread('MathBooks.png', 0)

image_fscs = fscs(image)
image_scs = saturated_contrast_stretching(image, (1,99))

[image_hist, bins] = np.histogram(image, bins=np.arange(0,numbins,1))
[image_fscs_hist, bins] = np.histogram(image_fscs, bins=np.arange(0,numbins,1))
[image_scs_hist, bins] = np.histogram(image_scs, bins=np.arange(0,numbins,1))

fig, plts = plt.subplots(1,3,figsize=(15,6))
plts[0].imshow(image, cmap='gray')
plts[1].imshow(image_scs, cmap='gray')
plts[2].plot(image_hist, color='blue', label=r"Original")
plts[2].plot(image_fscs_hist, color='red', label=r"Linear Constrast Stretching")
plts[2].plot(image_scs_hist, color='green', label=r"Saturated Constrast Stretching")
plts[2].set_xlim([0,255])
plts[2].legend()
# plt.savefig('/Users/abhijith/Desktop/TECHNOLOGIE/Courses/E9 241 Digital Signal Processing/Assignments/Assignment_2/Answers/figures/sat_stretch.eps', format='eps')

plt.show()


