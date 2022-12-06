#!/usr/bin/env python
# coding: utf-8

# # My Functions Library
# 
#

import numpy as np
import multiprocessing as mp
import collections.abc

def pool_init(shared_array_,srcimg, imgfilter):
    #shared_array_: is the shared read/write data, with lock. It is a vector (because the shared memory should be allocated as a vector
    #srcimg: is the original image
    #imgfilter is the filter which will be applied to the image and stor the results in the shared memory array
    
    #We defines the local process memory reference for shared memory space
    global shared_space
    #Here we define the numpy matrix handler
    global shared_matrix
    
    #Here, we will define the readonly memory data as global (the scope of this global variables is the local module)
    global image
    global my_filter
    
    #here, we initialize the global read only memory data
    image=srcimg
    my_filter=imgfilter
    size = image.shape
    
    #Assign the shared memory  to the local reference
    shared_space = shared_array_
    #Defines the numpy matrix reference to handle data, which will uses the shared memory buffer
    shared_matrix = tonumpyarray(shared_space).reshape(size)

def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)

def getPixel(row, col):
    # getPixel: is a helper function for grabbing pixels to be used in the filter calculations. It ensures that we use the border pixels when we would be outside the bounds
    # row: the row index of the pixel needed for applying a filter
    # col: the col index of the pixel needed for applying a filter
    global image
    
    (rows,cols,depth) = image.shape
    
    r = row
    c = col
    
    # If we are past the top border we want to use pixels from the top row
    if r < 0:
        r = 0
    # If we are past the bottom border we want to use pixels from the bottom row
    elif r >= rows:
        r = rows - 1
    # If we are past the top border we want to use pixels from the top row
    if c < 0:
        c = 0
    # If we are past the bottom border we want to use pixels from the bottom row
    elif c >= cols:
        c = cols - 1
    # Returns the pixel to be used in our filter equation
    return image[r,c,:]

def row_filter(row):
    global image
    global my_filter
    global shared_space
    
    (rows,cols,depth) = image.shape
    # print(image.shape)
    #defines the result vector, and set the initial value to 0
    frow=np.zeros((cols,depth))
    
    for col, pixel in enumerate(frow):
        # Create an array that contains the grid of pixels we will use with the filter
        li = []
        # Iterates from the rows below to the rows above the current row according to the shape of the filter
        # print(my_filter.shape)
        
        num_filter_rows = len(my_filter)
        num_filter_cols = len(my_filter[0])
        # # If there is just one 
        # if isinstance(my_filter[0], collections.abc.Sequence):
        #     num_filter_cols = len(my_filter[0])
        # else:
        #     num_filter_cols = len(my_filter
        for r in range(row - num_filter_rows//2, row + num_filter_rows//2 + 1):
            curr_row_li = []
            # Iterates from the columns below to the columns above the current column according to the shape of the filter
            for c in range(col - num_filter_cols//2, col + num_filter_cols//2 + 1):
                pixel_vals = getPixel(r,c)
                curr_row_li.append(pixel_vals)
            li.append(curr_row_li)
        arr = np.array(li)

        # Element wise multiply this array with our filter to calculate the product of these pixels with the filter
        productsMtrx = np.zeros(arr.shape)
        # Multiply each pixel in the array representing the pixels we will use in the filter calculation by the correct filter value
        for r in range(len(arr)):
            for c in range(len(arr[0])):
                productsMtrx[r][c] = arr[r][c] * my_filter[r][c]

        # Sum the products together to get the final pixel value
        frow[col] = np.sum(productsMtrx, axis=(0,1))
    
    with shared_space.get_lock():
        #while we are in this code block no ones, except this execution thread, can write in the shared memory
        shared_matrix[row,:,:]=frow
    return

def image_filter(image: np.array, filter_mask: np.ndarray, numprocessors: int, filtered_image: mp.Array):
    
    rows=range(image.shape[0])
    with mp.Pool(processes=numprocessors,initializer=pool_init,initargs=[filtered_image,image,filter_mask]) as p:
        p.map(row_filter,rows)

def filters_execution(image: np.array, filter_mask1: np.ndarray, filter_mask2: np.ndarray, numprocessors: int, filtered_image1: mp.Array, filtered_image2: mp.Array):
    # Divide number of available processors by 2 to allot half to each process
    numprocess = int(numprocessors/2)
    
    #defines both processes
    p1 = mp.Process(target=image_filter, args=(image,filter_mask1,numprocess,filtered_image1))
    p2 = mp.Process(target=image_filter, args=(image,filter_mask2,numprocess,filtered_image2))
    
    #fires both processes in parallel
    p1.start()
    p2.start()
    
    #Now, we have to wait until both parallel tasks complete
    p1.join()
    p2.join()
    return