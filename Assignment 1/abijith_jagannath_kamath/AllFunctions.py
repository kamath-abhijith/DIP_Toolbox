# This file should contain all the functions required by Wrapper.py

import sys
sys.path.insert(1,'abijith_jagannath_kamath/')

import time
import numpy as np

from pathlib import Path
from matplotlib import image as mpimg
# from union_find import UnionFind

def compute_hist(image_path: Path, num_bins: int) -> list:
    # bins_vec and freq_vec should contain values computed by custom function
    # bins_vec_lib and freq_vec_lib should contain values computed by python library function
    image = mpimg.imread(image_path,0)
    
    bins_vec = np.arange(0,1,1/num_bins)
    freq_vec = np.zeros(num_bins)
    for i in range(0,num_bins-1):
        freq_vec[i] = sum(sum(np.logical_and(image>=bins_vec[i], image<bins_vec[i+1])))

    [hist_from_numpy, bins_from_numpy] = np.histogram(image, bins=np.arange(0,1,1/num_bins))
    bins_vec_lib = bins_from_numpy
    freq_vec_lib = np.append(hist_from_numpy,[0])
    return [bins_vec, freq_vec, bins_vec_lib, freq_vec_lib]

def otsu_threshold(gray_image_path: Path) -> list:
    
    thr_w = thr_b = time_w = time_b = 0
    eps = 1e-10

    # Read image
    image = mpimg.imread(gray_image_path,0)
    [M,N] = image.shape
    threshold_limits=(50,200)

    # Compute histogram density
    bin_len = 255
    [hist, _] = np.histogram(image, bins=np.arange(0,1,1/bin_len))
    hist = np.hstack((hist, [0]))

    image_density = hist/(M*N)

    # Compute class probabilities
    omega0_t = np.cumsum(image_density)
    omega1_t = np.flip(np.cumsum(np.flip(image_density)))

    # Compute class means
    imagemean_k = np.arange(0,bin_len)*image_density

    mean0_t = np.cumsum(imagemean_k)/(omega0_t+eps)
    mean1_t = np.flip(np.cumsum(np.flip(imagemean_k)))/(omega1_t+eps)

    # Compute class variances
    imagevar_k = (np.arange(0,bin_len)**2)*image_density

    var0_t = np.cumsum(imagevar_k)/(omega0_t+eps) - mean0_t**2
    var1_t = np.flip(np.cumsum(np.flip(imagevar_k)))/(omega1_t+eps) - mean1_t**2

    # Minimising within class variance
    t_within_class = time.time()
    within_class_variance = omega0_t*var0_t + omega1_t*var1_t
    thr_w = np.argmin(within_class_variance[threshold_limits[0]:threshold_limits[1]])
    time_w = time.time() - t_within_class

    # Maximising between class variance (slightly faster)
    t_between_class = time.time()
    between_class_variance = omega0_t*omega1_t*(mean0_t-mean1_t)**2
    thr_b = np.argmax(between_class_variance[threshold_limits[0]:threshold_limits[1]])
    time_b = time.time() - t_between_class

    # Compute the binary image
    bin_image = (image>=thr_b/bin_len)*1

    return [thr_w, thr_b, time_w, time_b, bin_image]

def change_background(quote_image_path: Path, bg_image_path: Path) -> np.ndarray:
    
    _, _, _, _, bin_image = otsu_threshold(quote_image_path)
    back_image = mpimg.imread(bg_image_path,0)

    modified_image = back_image + bin_image
    return modified_image

def pixel_neighbours(image, connectivity, x, y):
    
    labels = set()

    if (connectivity==4) or (connectivity==8):
        if x > 0:
            west_neighbour = image[y,x-1]
            if west_neighbour > 0:
                labels.add(west_neighbour)
        
        if y > 0:
            north_neighbour = image[y-1,x]
            if north_neighbour > 0:
                labels.add(north_neighbour)

        if connectivity == 8:
                if x > 0 and y > 0:
                    northwest_neighbour = image[y-1,x-1]
                    if northwest_neighbour > 0:
                        labels.add(northwest_neighbour)

                if y > 0 and x < len(image[y]) - 1:
                    northeast_neighbour = image[y-1,x+1]
                    if northeast_neighbour > 0:
                        labels.add(northeast_neighbour)
    
    else:
        print("Connectivity type not found.")
        
    return labels

def count_connected_components(gray_image_path: Path) -> int:
    
    from union_find import UnionFind

    image = mpimg.imread(gray_image_path, 0)

    thr, _, _, _, _ = otsu_threshold(gray_image_path)
    bool_input_image = (image<=(thr/255))

    image_width = len(bool_input_image[0])
    image_height = len(bool_input_image)
    
    labelled_image = np.zeros((image_height, image_width), dtype=np.int16)
    uf = UnionFind()
    current_label = 1

    # First Pass
    for y, row in enumerate(bool_input_image):
        for x, pixel in enumerate(row):
			
            if pixel == False:
                pass
            else:
                # Find neighbours
                labels = pixel_neighbours(labelled_image, 4, x, y)
                
                if not labels:
					# No connecting pixels
                    labelled_image[y,x] = current_label
                    uf.MakeSet(current_label)
                    current_label = current_label + 1
                    
                else:
                    # Already in a connected set
                    smallest_label = min(labels)
                    labelled_image[y,x] = smallest_label

					# Conflict
                    if len(labels) > 1:
                        for label in labels:
                            uf.Union(uf.GetNode(smallest_label), uf.GetNode(label))


	# Second Pass
    final_labels = {}
    new_label_number = 1
    
    for y, row in enumerate(labelled_image):
        for x, pixel_value in enumerate(row):
            
            if pixel_value > 0: # Labelled pixel
				# Find correction
                new_label = uf.Find(uf.GetNode(pixel_value)).value
                labelled_image[y,x] = new_label

				# Conflict
                if new_label not in final_labels:
                    final_labels[new_label] = new_label_number
                    new_label_number = new_label_number + 1


	# Third Pass
    for y, row in enumerate(labelled_image):
        for x, pixel_value in enumerate(row):
            if pixel_value > 0: # Labelled pixel
                labelled_image[y,x] = final_labels[pixel_value]

    num_characters = np.max(labelled_image)
    return num_characters

def erosion(image, kernel):
    
    eroded_image = image.copy()

    y_window = image.shape[0] - kernel.shape[0]
    x_window = image.shape[1] - kernel.shape[1]

    y_pos = 0
    while y_pos <= y_window:
        x_pos = 0

        while x_pos <= x_window:
            flag = False

            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    if kernel[i][j] == 1:
                        if image[y_pos+i][x_pos+j] == 0:
                            flag = True
                            break
                
                if flag:
                    eroded_image[y_pos, x_pos] = 0
                    break

            x_pos += 1
        y_pos += 1

    return eroded_image

def dilation(image, kernel):
    
    dilated_image = image.copy()

    y_window = image.shape[0] - kernel.shape[0]
    x_window = image.shape[1] - kernel.shape[1]

    y_pos = 0
    while y_pos <= y_window:
        x_pos = 0

        while x_pos <= x_window:
            flag = False

            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    if kernel[i][j] == 1:
                        if image[y_pos+i][x_pos+j] == 1:
                            flag = True
                            break
                
                if flag:
                    dilated_image[y_pos, x_pos] = 1
                    break

            x_pos += 1
        y_pos += 1
        
    return dilated_image

def binary_morphology(gray_image_path: Path) -> np.ndarray:
    _, _, _, _, bin_image = otsu_threshold(gray_image_path)
    
    kernel = np.array ([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype = np.uint8)

    level1 = dilation(bin_image, kernel)
    level2 = erosion(level1, kernel)
    cleaned_image = dilation(level2, kernel)
    
    return cleaned_image

def connected_component_labels(bool_input_image):
    
    from union_find import UnionFind
    
    image_width = len(bool_input_image[0])
    image_height = len(bool_input_image)
    
    labelled_image = np.zeros((image_height, image_width), dtype=np.int16)
    uf = UnionFind()
    current_label = 1

    # First Pass
    for y, row in enumerate(bool_input_image):
        for x, pixel in enumerate(row):
			
            if pixel == False:
                pass
            else:
                # Find neighbours
                labels = pixel_neighbours(labelled_image, 4, x, y)
                
                if not labels:
					# No connecting pixels
                    labelled_image[y,x] = current_label
                    uf.MakeSet(current_label)
                    current_label = current_label + 1
                    
                else:
                    # Already in a connected set
                    smallest_label = min(labels)
                    labelled_image[y,x] = smallest_label

					# Conflict
                    if len(labels) > 1:
                        for label in labels:
                            uf.Union(uf.GetNode(smallest_label), uf.GetNode(label))


	# Second Pass
    final_labels = {}
    new_label_number = 1
    
    for y, row in enumerate(labelled_image):
        for x, pixel_value in enumerate(row):
            
            if pixel_value > 0: # Labelled pixel
				# Find correction
                new_label = uf.Find(uf.GetNode(pixel_value)).value
                labelled_image[y,x] = new_label

				# Conflict
                if new_label not in final_labels:
                    final_labels[new_label] = new_label_number
                    new_label_number = new_label_number + 1


	# Third Pass
    for y, row in enumerate(labelled_image):
        for x, pixel_value in enumerate(row):
            if pixel_value > 0: # Labelled pixel
                labelled_image[y,x] = final_labels[pixel_value]

    return labelled_image

def count_mser_components(gray_image_path: Path) -> list:
    
    from union_find import UnionFind

    thr, _, _, _, otsu_binary_image = otsu_threshold(gray_image_path)
    otsu_binary_image = np.logical_not(otsu_binary_image)
    labelled_otsu_image = connected_component_labels(otsu_binary_image)
    num_otsu_components = np.max(labelled_otsu_image)

    image = mpimg.imread(gray_image_path, 0)

    Lthr = 150
    Uthr = 160
    eps = 1
    delta = 0.2
    critical_thr = []
    component_size_difference = dict()
    for thr in range(Lthr,Uthr):
        bin_image_thr = np.logical_not(image<=(thr/255))
        bin_image_threps = np.logical_not(image<=((thr-eps)/255))

        labelled_image_thr = connected_component_labels(bin_image_thr)
        labelled_image_threps = connected_component_labels(bin_image_threps)
        component_size_difference[thr] = dict()
        for i in range(np.max(labelled_image_thr)):
            component_size_difference[thr][i] = sum(sum(labelled_image_thr==(i+1))) - sum(sum(labelled_image_threps==(i+1)))
            if (component_size_difference[thr][i]<=delta):
                critical_thr.append(thr)
                break

    thr_mser_min = np.min(critical_thr)
    thr_mser_max = np.max(critical_thr)
    mser_binary_image = np.logical_not((image>=thr_mser_min/255)*1) + (image>=thr_mser_max/255)*1
    
    labelled_mser_image = connected_component_labels(mser_binary_image)
    num_mser_components = np.max(labelled_mser_image)

    return [mser_binary_image, otsu_binary_image, num_mser_components, num_otsu_components]
