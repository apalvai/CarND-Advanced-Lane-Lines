import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_image_and_object_points(nx, ny):

    # image file names
    image_filenames = glob.glob('../camera_cal/calibration*.jpg')
    
    # object and image points
    obj_points = []
    img_points = []
    
    # prepare object points
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    for filename in image_filenames:

        image = mpimg.imread(filename)

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)

            # draw and display the corners
            # img_with_corners = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            # plt.show(img_with_corners)
        
    return obj_points, img_points

def calibrate_and_undistort(image, obj_points, img_points):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    undistorted_img = cv2.undistort(image, mtx, dist, None, mtx)
    return undistorted_img

def undistort(image):
    obj_points, img_points = get_image_and_object_points(9, 6)
    undistorted_img = calibrate_and_undistort(image, obj_points, img_points)
    return undistorted_img

def test():
    test_image = cv2.imread('../test_images/straight_lines2.jpg')
    undistorted_img = undistort(test_image)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(test_image)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undistorted_img)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()

test()
