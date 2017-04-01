import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

from camera_calibration import undistort

def get_source_points():
    # read one of the camera calibration chessboard images
    image_filenames = glob.glob('../camera_cal/calibration*.jpg')
    
    nx = 9
    ny = 6
    
    corners = []
    
    for filename in image_filenames:
        
        image = mpimg.imread(filename)
        
        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # find chessboard corners
        ret, chessboard_corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret == True:
            print('found corners for filename: ', filename)
            corners = chessboard_corners
            break
    
    # identify source points
    src_points = np.float32([corners[0][0], corners[nx-1][0], corners[((nx)*(ny-1))][0], corners[nx*ny-1][0]])
    
    return src_points

def get_destination_points(img, src_points):
    
    nx = 9
    ny = 6
    
    dx = img.shape[1]/(nx+1)
    dy = img.shape[0]/(ny+1)
    
    mul = 2/3
    img_size = (img.shape[1], img.shape[0])
    
    dst = np.float32([[dx*mul, dy*mul], [img_size[0]-dx*mul, dy*mul], [dx*mul, img_size[1]-dy*mul], [img_size[0]-dx*mul, img_size[1]-dy*mul]])
    
    return dst

# Apply perspective transform to unwarp an image
def warp(image, src_points):
    
    dst_points = get_destination_points(image, src_points)
    print('dst ponts: ', dst_points)
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    image_size = (image.shape[1], image.shape[0])
    
    warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    
    return warped_image

def test():
    # read test image
    test_image = cv2.imread('../camera_cal/calibration10.jpg')
    
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
