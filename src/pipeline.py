import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from camera_calibration import undistort
from perspective_transform import warp, get_source_points
from gradient import abs_sobel_thresh, mag_thresh, dir_threshold, GRAD_ABS_THRESH_MIN, GRAD_ABS_THRESH_MAX, GRAD_DIR_THRESH_MIN, GRAD_DIR_THRESH_MAX
from color import hls_color_binary, rgb_color_binary, gray_binary

def apply_gradient_and_color_threshold(image, s_thresh=(170, 255), sx_thresh=(20, 100)):

    image = np.copy(image)
    
    grad_binary_x = abs_sobel_thresh(image, orient='x', thresh_min=sx_thresh[0], thresh_max=sx_thresh[1])
    color_binary = hls_color_binary(image, thresh_min=s_thresh[0], thresh_max=s_thresh[1], color_channel='s')
    
    combined_binary = np.zeros_like(color_binary)
    combined_binary[(grad_binary_x == 1) | (color_binary == 1)] = 1
    
    return combined_binary

def test():
    image = mpimg.imread('../test_images/test3.jpg')
    
    undistorted_image = undistort(image)
    
    thresholded_image = apply_gradient_and_color_threshold(undistorted_image)
    
    src_points = get_source_points()
    result = warp(thresholded_image, src_points)
    
    # Plot the result
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
    f.tight_layout()
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=15)
    
    ax2.imshow(undistorted_image, cmap='gray')
    ax2.set_title('Undistorted', fontsize=15)
    
    ax3.imshow(thresholded_image, cmap='gray')
    ax3.set_title('Thresholded', fontsize=15)
    
    ax4.imshow(result, cmap='gray')
    ax4.set_title('Unwarped', fontsize=15)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
    plt.show()

test()
