import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from camera_calibration import get_calibration_metrics, Calibration
from perspective_transform import warp, get_source_points, get_destination_points
from gradient import combined_gradient_threshold
from color import hls_color_binary, rgb_color_binary, gray_binary

from Line import Line

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Setup camera calibration
calibration = Calibration()

# Set up lines for left and right
left_line = Line()
right_line = Line()

def select_region_of_interest(image):
    # create a mask
    mask = np.zeros_like(image)
    
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    # Defining vertices for region of interest
    imshape = image.shape
    left_bottom = (100, imshape[0])
    left_top = (600, 410)
    right_top = (690, 410)
    right_bottom = (imshape[1]-50, imshape[0])
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def apply_gradient_and_color_threshold(image, s_thresh=(170, 255)):

    image = np.copy(image)
    
    combined_gradient_binary = combined_gradient_threshold(image)
    color_binary = hls_color_binary(image, thresh_min=s_thresh[0], thresh_max=s_thresh[1], color_channel='s')
    
    combined_binary = np.zeros_like(color_binary)
    combined_binary[(combined_gradient_binary == 1) | (color_binary == 1)] = 1
    
    # select region of interest
    result = select_region_of_interest(combined_binary)
    
    return result

def process_image(image):
    
    # calibrate camera and undistort
    if  calibration.mtx == None:
        print ('computing camera calibration metrics...')
        ret, mtx, dist, rvecs, tvecs = get_calibration_metrics(image)
        calibration.mtx = mtx
        calibration.dist = dist
        calibration.rvecs = rvecs
        calibration.tvecs = tvecs
    
    undistorted_image = cv2.undistort(image, calibration.mtx, calibration.dist, None, calibration.mtx)
    
    # apply gradient and color thresholds
    thresholded_image = apply_gradient_and_color_threshold(undistorted_image)
    
    # apply perspective transform
    src_points = get_source_points()
    result = warp(thresholded_image, src_points)
    
    # Plot the result
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#    f.tight_layout()
#    
#    ax1.imshow(image)
#    ax1.set_title('Original Image', fontsize=15)
#    
#    ax2.imshow(result, cmap='gray')
#    ax2.set_title('Output', fontsize=15)
#    
#    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.)
#    plt.show()

    return result

def get_line_pixels_and_fit(binary_warped, left_fit=None, right_fit=None):
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the width of the windows +/- margin
    margin = 100
    
    if (left_fit == None) | (right_fit == None):
        print('applying sliding window to detect lane lines...')
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    else:

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
                          
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each in pixels
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
#    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#    
#    # Create an image to draw on and an image to show the selection window
#    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
#    window_img = np.zeros_like(out_img)
#    
#    # Color in left and right line pixels
#    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#    
#    # Generate a polygon to illustrate the search window area
#    # And recast the x and y points into usable format for cv2.fillPoly()
#    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
#    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
#    left_line_pts = np.hstack((left_line_window1, left_line_window2))
#    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
#    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
#    right_line_pts = np.hstack((right_line_window1, right_line_window2))
#    
#    # Draw the lane onto the warped blank image
#    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
#    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
#    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
#    plt.imshow(result)
#    plt.plot(left_fitx, ploty, color='yellow')
#    plt.plot(right_fitx, ploty, color='yellow')
#    plt.xlim(0, 1280)
#    plt.ylim(720, 0)
#    plt.show()

    return ploty, leftx, lefty, rightx, righty, left_fit, right_fit

def radius_of_curvature_in_pixels(y_eval, left_fit, right_fit):
    
    # Define y-value where we want radius of curvature
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # print('radius of curvature in pixels at: ', y_eval, left_curverad, right_curverad)
    
    return left_curverad, right_curverad

def radius_of_curvature_in_meters(y_eval, leftx, lefty, rightx, righty, left_fit, right_fit):
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    # print('radius of curvature in meters at: ', y_eval, left_curverad, right_curverad)
    
    return left_curverad, right_curverad

def get_position(image_shape, pts):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    position = image_shape[1]/2
    left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 700)][:,1])
    right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 700)][:,1])
    center = (left + right)/2
    return (position - center)*xm_per_pix

def draw_poly(image, result, yvals, leftx, lefty, rightx, righty, left_fit, right_fit, curvature):
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(result).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Apply inverse perspective tansform
    src_points = get_source_points()
    dst_points = get_destination_points(image)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    
    undistorted_image = cv2.undistort(image, calibration.mtx, calibration.dist, None, calibration.mtx)
    
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
    
    font = cv2.FONT_HERSHEY_PLAIN
    
    # Show curvature on an image
    text = "Radius of Curvature: {} m".format(int(curvature))
    cv2.putText(result, text, (50, 100), font, 3, (255, 255, 255), 2)

    # Show position on an image
#    pts = np.argwhere(newwarp[:,:,1])
#    position = get_position(image.shape, pts)
#    if position < 0:
#        text = "Vehicle is {:.2f} m left of center".format(-position)
#    else:
#        text = "Vehicle is {:.2f} m right of center".format(position)
#    cv2.putText(result, text, (50, 150), font, 3, (255, 255, 255), 2)

    return result

def find_lane_lines(image):
    # process image
    result = process_image(image)
    
    # fit a polynomial of 2nd degree for lane lines based on sliding window technique
    left_fit = None
    if len(left_line.current_fit) > 0:
        left_fit = left_line.current_fit[0]
    
    right_fit = None
    if len(right_line.current_fit) > 0:
        right_fit = right_line.current_fit[0]

    yvals, leftx, lefty, rightx, righty, left_fit, right_fit = get_line_pixels_and_fit(result, left_fit, right_fit)
    
    left_line.current_fit = [left_fit]
    right_line.current_fit = [right_fit]
    
    # measure radius of curvature
    y_eval = np.max(yvals)
    radius_of_curvature_in_pixels(y_eval, left_fit, right_fit)
    left_curv, right_curv = radius_of_curvature_in_meters(y_eval, leftx, lefty, rightx, righty, left_fit, right_fit)
    
    # draw the polygon
    result = draw_poly(image, result, yvals, leftx, lefty, rightx, righty, left_fit, right_fit, left_curv)

    return result

def test():
    
    for i in range(1, 7):
        filename = '../test_images/test{}.jpg'.format(i)
        
        image = mpimg.imread(filename)
        
        # find lane lines
        result = find_lane_lines(image)
        
        plt.imshow(result)
        plt.show()

#test()

from moviepy.editor import VideoFileClip

white_output = '../white.mp4'
clip1 = VideoFileClip('../project_video.mp4')
white_clip = clip1.fl_image(find_lane_lines) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

