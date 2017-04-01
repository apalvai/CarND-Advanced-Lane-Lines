import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for adjusting the min and max thresholds for x/y gradients
GRAD_ABS_THRESH_MIN = 20
GRAD_ABS_THRESH_MAX = 100

GRAD_MAG_THRESH_MIN = 30
GRAD_MAG_THRESH_MAX = 100

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

    #Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the gradient in x and y separately
    sobel_gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    # Calculate the magnitude
    abs_sobel_gradient = np.sqrt(sobel_gradient_x**2 + sobel_gradient_y**2)
    
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel_gradient/np.max(abs_sobel_gradient))
    
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    return binary_output

def test():
    
    image = mpimg.imread('../test_images/test1.jpg')
    grad_binary_x = abs_sobel_thresh(image, orient='x', thresh_min=GRAD_ABS_THRESH_MIN, thresh_max=GRAD_ABS_THRESH_MAX)
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(GRAD_MAG_THRESH_MIN, GRAD_MAG_THRESH_MAX))
    
    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(grad_binary_x, cmap='gray')
    ax2.set_title('Thresholded Gradient (X)', fontsize=15)
    ax3.imshow(mag_binary, cmap='gray')
    ax3.set_title('Gradient Maginitude', fontsize=15)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
    plt.show()

test()
