import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def gray_binary(image, thresh=(0, 255)):
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # create binary and apply thresholds
    gray_binary = np.zeros_like(gray)
    gray_binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    
    return gray_binary

def rgb_color_binary(image, color_channel='r', thresh=(0, 255)):
    
    # read the relevant color channel (assuming RGB channels as the order)
    channel = []
    if color_channel == 'r':
        channel = image[:, :, 0]
    elif color_channel == 'g':
        channel = image[:, :, 1]
    elif color_channel == 'b':
        channel = image[:, :, 2]

    # Create a binary and apply color thresholds
    color_binary = np.zeros_like(channel)
    color_binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    
    return color_binary

def hls_color_binary(image, thresh_min, thresh_max, color_channel='s'):
    
    # convert the rgb image to hls image
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # read the relevant color channel (assuming HLS channels as the order)
    channel = []
    if color_channel == 'h':
        channel = hls[:, :, 0]
    elif color_channel == 'l':
        channel = hls[:, :, 1]
    elif color_channel == 's':
        channel = hls[:, :, 2]
    
    # Create a binary and apply color thresholds
    color_binary = np.zeros_like(channel)
    color_binary[(channel > thresh_min) & (channel <= thresh_max)] = 1
    
    return color_binary

def test():
    # read test image
    image = mpimg.imread('../test_images/test6.jpg')
    
    gray = gray_binary(image, thresh=(180, 255))
    r_binary = rgb_color_binary(image, color_channel='r', thresh=(200, 255))
    s_binary = hls_color_binary(image, thresh_min=90, thresh_max=255, color_channel='s')
    
    # Plot the result
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
    f.tight_layout()
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=15)
    
    ax2.imshow(gray, cmap='gray')
    ax2.set_title('Gray', fontsize=15)
    
    ax3.imshow(r_binary, cmap='gray')
    ax3.set_title('Red (RGB)', fontsize=15)
    
    ax4.imshow(s_binary, cmap='gray')
    ax4.set_title('Saturation (HLS)', fontsize=15)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
    plt.show()

#test()
