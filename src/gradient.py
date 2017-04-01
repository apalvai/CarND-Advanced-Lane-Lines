import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for adjusting the min and max thresholds for x/y gradients
GRAD_MIN_THRESHOLD_XY = 20
GRAD_MAX_THRESHOLD_XY = 100

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Convert to grayscale (assuming the image has RGB channels)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel_gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel_gradient = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    # Take the absolute value of the derivative or gradient
    abs_sobel_gradient = np.absolute(sobel_gradient)
    
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel_gradient/np.max(abs_sobel_gradient))

    binary_output = np.zeros_like(scaled_sobel)

    #Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return binary_output

def test():
    
    image = mpimg.imread('../test_images/test1.jpg')
    grad_binary_x = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
    grad_binary_y = abs_sobel_thresh(image, orient='y', thresh_min=20, thresh_max=100)
    
    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(grad_binary_x, cmap='gray')
    ax2.set_title('Thresholded Gradient (X)', fontsize=15)
    ax3.imshow(grad_binary_y, cmap='gray')
    ax3.set_title('Thresholded Gradient (Y)', fontsize=15)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
    plt.show()

test()
