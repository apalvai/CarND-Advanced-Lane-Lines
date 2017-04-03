import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

from camera_calibration import undistort

# http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

'''
NOTE: Order of points should be [top_left, top_right, bottom_right, bottom_left]
'''

def get_source_points():
    
    return np.float32([[590, 450], [700, 450], [1130, 700], [200, 700]])

def get_destination_points(img):
    
    offset = 300
    img_size = (img.shape[1], img.shape[0])
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
    
    return dst

# Apply perspective transform to unwarp an image
def warp(image, src_points):
    
    dst_points = get_destination_points(image)
    # print('dst ponts: ', dst_points)
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    image_size = (image.shape[1], image.shape[0])
    
    warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    
    return warped_image

def test():
    # read test image
    test_image = cv2.imread('../test_images/test5.jpg')
    
    # undistort the image
    undistorted_image = undistort(test_image)
    
    # get source points from camera calibration images
    src_points = get_source_points()
    print('src ponts: ', src_points)
    
    # unwarp the image
    unwarped_image = warp(undistorted_image, src_points)
    
    # plot
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(test_image)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(unwarped_image)
    ax2.set_title('Unwarped Image', fontsize=30)
    plt.show()

#test()
