import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys
from moviepy.editor import VideoFileClip
import imageio


# find chess board corners
def corr_coef(path, file, nx=9, ny=6):
    fname = path + file
    img = cv2.imread(fname)
    # converts to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # finds the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    return ret, corners


# unwarp perspective of an image
def corners_unwarp(img, src, dest, mtx, dist):
    # Undistorts using mtx and dist
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Converts to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # uses cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dest)
    # Minv - inverted matrix
    Minv = cv2.getPerspectiveTransform(dest, src)
    img_size = (gray.shape[1], gray.shape[0])
    # uses cv2.warpPerspective() to warp the image to a top-down view
    warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


# finds calibration parameters for camera
def cal_param(path, nx=9, ny=6):
    # prepares object points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2d points in image plane

    # prepares object points, like (0,0,0),(1,0,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates

    # makes a list of calibration images
    edge_found = 0
    edge_not_found = 0
    for file in os.listdir(path):
        fname = path + file
        img = cv2.imread(fname)
        ret, corners = corr_coef(path, file, nx=9, ny=6)
        # If found, append corners to the lists
        if ret == True:
            edge_found += 1
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            edge_not_found += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calibration parameters for camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


# calculates more or less equal number of rows and cols for plots with multiple sublots
def nrows_ncols(n):
    if (int(np.sqrt(n))) ** 2 == n:
        n_col = n_row = np.sqrt(n)
    else:
        n_col = n // int(np.sqrt(n))
        n_row = n // n_col + n % n_col
    return n_row, n_col


# plots multiple images on the same plt.figure(), camera was not calibrated yet, this function is not neccessary for
# final pipeline
# all images will be ploted in pairs: original images from "files" and images with applied func(*args) on it
def multiple_plot_uncalibrated(path, files, func, title1='Original image', title2='Modified', *args):
    fig = plt.figure()
    n = len(files)  # number of pictures to show
    n_row, n_col = nrows_ncols(
        2 * n)  # multiplies by two, because it should show original image + image with applied func(*args) on it
    for i in range(n):
        a = fig.add_subplot(n_row, n_col, 2 * i + 1)
        img = mpimg.imread(path + files[i])
        plt.imshow(img)
        a.set_title(title1, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
        a = fig.add_subplot(n_row, n_col, 2 * i + 2)
        a.set_title(title2, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
        dst = func(img, *args)
        plt.imshow(dst)
    fig.savefig('output_images/' + title2 + '.jpg')
    return fig


# plots multiple images on the same plt.figure()
# all images will be ploted in pairs: original images from "files" and images with applied func(*args) on it
def multiple_plot(path, files, func, title1='Original image', title2='Modified', *args):
    ret, mtx, dist, rvecs, tvecs = cal_param('camera_cal/')
    fig = plt.figure()
    n = len(files)  # number of pictures to show
    n_row, n_col = nrows_ncols(
        2 * n)  # multiplies by two, because it will show original + image with applied func() on it
    for i in range(n):  # images calibrationXX, where XX = [2,4] are the most distorted from camera images
        a = fig.add_subplot(n_row, n_col, 2 * i + 1)
        img = mpimg.imread(path + files[i])
        plt.imshow(img)
        a.set_title(title1, fontsize=10)
        a = fig.add_subplot(n_row, n_col, 2 * i + 2)
        a.set_title(title2, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
        img = cv2.undistort(img, mtx, dist, None, mtx)
        dst = func(img, *args)
        a.set_xticks([])
        a.set_yticks([])
        plt.imshow(dst)
        fig.savefig('output_images/' + title2 + '.jpg')
    return fig


# checks on several images how the algoritm for calibrate camera + unwarp works
def undistort(path, files, mtx, dist, title2, nx=9, ny=6):
    fig = plt.figure()
    n = len(files)
    n_row, n_col = nrows_ncols(2 * n)
    for i in range(2):
        ret, corners = corr_coef(path, files[i], nx=9, ny=6)
        img = cv2.imread(path + files[i])
        # defines 4 source points src = np.float32([[,],[,],[,],[,]])
        src = np.float32([corners[0][0], corners[nx - 1][0], corners[nx * ny - 1][0], corners[nx * (ny - 1)][0]])
        # defines 4 destination points dst = np.float32([[,],[,],[,],[,]])
        x = img.shape[1]
        y = img.shape[0]
        dst = np.float32([[x // nx, y // ny], [x // nx * (nx - 1), y // ny], [x // nx * (nx - 1), y // ny * (ny - 1)],
                          [x // nx, y // ny * (ny - 1)]])
        unwarped = corners_unwarp(img, src, dst, mtx, dist)
        a = fig.add_subplot(n_row, n_col, 2 * i + 1)
        a.imshow(img)
        a.set_title('Original Image')
        a = fig.add_subplot(n_row, n_col, 2 * i + 2)
        a.imshow(unwarped)
        a.set_title(title2)
    fig.savefig('output_images/' + title2 + '.jpg')


# applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, mag_thresh=(0, 255)):
    # applies the following steps to img
    # converts to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # takes the derivative in x or y given orient = 'x' or 'y'
    # takes the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel))
    # scales to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # returns this mask as binary_output image
    return binary_output


# defines a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh_func(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # applies the following steps to img
    # converts to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # takes the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    # calculates the magnitude
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # scales to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(mag) / 255
    mag = (mag / scale_factor).astype(np.uint8)
    # create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(mag)
    binary_output[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 1
    # return this mask as your binary_output image
    return binary_output


# deletes everything in images besides the retion of interest
def region_of_interest(img, vertices=0):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # defines a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on the image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2), vertices=0):
    # applies the following steps to img
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # takes the absolute value of the x and y gradients
    ##abs_x = np.absolute(sobelx)
    ##abs_y = np.absolute(sobely)
    # np.arctan2(abs_sobely, abs_sobelx) calculates the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # creates a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # returns this mask as your binary_output image
    binary_output = region_of_interest(binary_output, vertices)
    return binary_output


# combimes abs_sobel_thresh() with dir_threshold() and masks everything besides the region of interest
def combined_sober(img, sobel_kernel=3, mag_thresh=(0, 255), thresh=(0, np.pi / 2), blur_kernel=0.0, vertices=0):
    # if blur_kernel!=(0,0):
    #     img = cv2.GaussianBlur(img, blur_kernel, 0)
    gradx = abs_sobel_thresh(img, 'x', sobel_kernel, mag_thresh)
    grady = abs_sobel_thresh(img, 'y', sobel_kernel, mag_thresh)
    mag_binary = mag_thresh_func(img, sobel_kernel, mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel, thresh, vertices)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # if blur_kernel!=(0,0):
    #     combined = cv2.GaussianBlur(combined, blur_kernel, 0)
    combined = region_of_interest(combined, vertices)
    return combined


# creates bitmap image from saturation channel of HLS images representation
def saturation_extraction(img, thresh_s=(90, 255), vertices=0):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh_s[0]) & (S <= thresh_s[1])] = 1
    binary = region_of_interest(binary, vertices)

    return binary


# This returns a stack of the two binary images, whose components are sobel transformation of an image and
# saturation channel from HLS with applied thresholds
def stack_sobel_saturation(img, sobel_kernel=3, mag_thresh=(20, 100), s_thresh=(170, 255), output='color'):
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    # Threshold x gradient
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # Threshold color channel
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    if output == 'color':
        return color_binary
    else:
        return combined_binary


# transforms image to bird eye view
def transform_image(img, src=np.float32([[200, 680], [590, 451], [690, 451], [1042, 666]]),
                    dest=np.float32([[180, 720], [200, 0], [1000, 0], [1000, 720]])):
    # uses cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dest)
    # uses cv2.warpPerspective() to warp your image to a top-down view
    Minv = cv2.getPerspectiveTransform(dest, src)
    img_size = (img.shape[1], img.shape[0])
    unwarped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return unwarped


from collections import deque

frame_buffer_left = deque()  # buffer for polynomial coefficients for left lane to avoid outbursts
frame_buffer_right = deque()  # buffer for polynomial coefficients for right lane to avoid outbursts
frame_curvature_left = deque()  # buffer for curvature for left lane to avoid outbursts
frame_curvature_right = deque()  # buffer for curvature for right lane to avoid outbursts


# mask for yellow lines
def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # lower = np.array([20,60,60])
    # upper = np.array([38,174, 250])
    lower = np.array([20, 60, 60])
    upper = np.array([38, 174, 250])

    mask = cv2.inRange(hsv, lower, upper)

    return mask


# mask for white lines
def select_white(image):
    lower = np.array([202, 202, 202])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    return mask


def yellow_white_sobel(img):
    mask_yellow = select_yellow(img)  # mask for yellow lines
    mask_white = select_white(img)  # mask for white lines
    img_sobel = stack_sobel_saturation(img, output='black')
    mask = np.zeros_like(img)
    mask[(mask_yellow > 0) | (mask_white > 0) | (img_sobel > 0)] = 1
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.dilate(mask, kernel, iterations = 2)
    # if output == 'mask':
    #     return mask
    # else:
    img_masked = img * mask
    return img_masked


# finds lines
def finding_lines(img, sobel_kernel=3, mag_thresh=(35, 100), thresh_s=(170, 255), output=0):
    # output=0 - outputs plt.fig with a plot with found on it lines
    # output=1 - returns polynomial coefficients for single images (not video frames), frame_buffers are
    #  not used during caclulation of coefficients
    # output=2 - outputs polynom's coefficients for static image
    # output=3  function skippes sliding windows). it means we know already where to look
    global frame_buffer_left  # collections.deque() for several last values of polynomial coefficients
    global frame_buffer_right  # collections.deque() for several last values of polynomial coefficients
    global frame_curvature_left  # collections.deque() for several last values of curvature
    global frame_curvature_right  # collections.deque() for several last values of curvature

    img = transform_image(img)
    binary_warped = stack_sobel_saturation(img, sobel_kernel, mag_thresh, thresh_s, 'bw')
    binary_warped = np.zeros_like(binary_warped)
    img_masked = yellow_white_sobel(img)
    binary_warped[(img_masked[:, :, 0] > 0) | (img_masked[:, :, 1] > 0) | (img_masked[:, :, 2] > 0)] = 1

    # takes a histogram of the bottom half of the 'binary_warped' image
    histogram = np.sum(binary_warped[3 * int(binary_warped.shape[0] / 4):, :], axis=0)
    # create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # finds the peak of the left and right halves of the histogram
    # these will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 50
    # sets height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # set the width of the windows +/- margin
    margin = 50
    # set minimum number of pixels found to recenter window
    minpix = 25
    # creates empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # steps through the windows one by one
    for window in range(nwindows):
        # identifies window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # draws the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # appends these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # if you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # concatenates the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # extracts left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # fits a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # generates x and y values for plotting
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if output == 0:  # image with wide polylinomial drawn on lanes
        return out_img
    elif output == 1:  # return polynomial coefficients for single images (not video frames), frame_buffers are
        # not used during caclulation of coefficients
        return left_fit, right_fit
    elif output == 2:  # return polynomial coefficients for single images from videos, frame_buffers are
        # used during caclulation of coefficients
        nframes = 2  # number of frames for frame_buffers to use
        margin = (0.5, 0.35, 4, 40)  ## deviation margin between last frame and last {n}frames
        #  {curvature_left, curvature_right, \
        #  {curvature_left/curvature_rigth or curvature_rigth/curvature_left,
        #  offset between old line and new line(if offest bigger than {40} values from previous
        #  video frame used)}} \

        curvature = calc_curvature(img)
        if len(frame_buffer_left) == 0:
            for i in range(nframes):
                frame_buffer_left.append(left_fit)
                frame_buffer_right.append(right_fit)
                frame_curvature_left.append(curvature[0])
                frame_curvature_right.append(curvature[1])
        ploty = binary_warped.shape[0]  # y of lowest polynomial line on the screen, \
        # left_fitx and left_fitx_previous appropriate x values for ploty \
        # in current and previous video frame
        left_fitx = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
        left_fitx_previous = frame_buffer_left[-1][0] * (ploty ** 2) + frame_buffer_left[-1][1] * ploty + \
                             frame_buffer_left[-1][2]
        if (1 + margin[0] > curvature[0] / frame_curvature_left[-1] > 1 - margin[0]) and (curvature[0] > 1000 or \
                                                                                                              1 /
                                                                                                              margin[
                                                                                                                  2] <
                                                                                                              curvature[
                                                                                                                  0] /
                                                                                                              curvature[
                                                                                                                  1] <
                                                                                                      margin[
                                                                                                          2]) and abs(
                    left_fitx - left_fitx_previous) < margin[3]:
            frame_buffer_left.popleft()
            frame_buffer_left.append(left_fit)
            frame_curvature_left.popleft()
            frame_curvature_left.append(curvature[0])
        else:
            left_fit = frame_buffer_left[-1]
        right_fitx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]
        right_fitx_previous = frame_buffer_right[-1][0] * (ploty ** 2) + frame_buffer_right[-1][1] * ploty + \
                              frame_buffer_right[-1][2]
        if (1 + margin[1] > curvature[1] / frame_curvature_right[-1] > 1 - margin[1]) and (curvature[1] > 1000 or \
                                                                                                               1 /
                                                                                                               margin[
                                                                                                                   2] <
                                                                                                               curvature[
                                                                                                                   1] /
                                                                                                               curvature[
                                                                                                                   0] <
                                                                                                       margin[
                                                                                                           2]) and abs(
                    right_fitx - right_fitx_previous) < margin[3]:
            frame_buffer_right.popleft()
            frame_buffer_right.append(right_fit)
            frame_curvature_right.popleft()
            frame_curvature_right.append(curvature[0])

        else:
            right_fit = frame_buffer_right[-1]
        return left_fit, right_fit

    elif output == 3:
        return finding_lines_continue(binary_warped, left_fit, right_fit)

    elif output == 3:
        return finding_lines_continue(binary_warped, left_fit, right_fit)


# function used, if skipped function finding_lines(aka sliding windows). it means we know already where to look
# for line's starting points
def finding_lines_continue(binary_warped, left_fit, right_fit):
    global frame_buffer_left  # collections.deque() for several last values of polynomial coefficients
    global frame_buffer_right  # collections.deque() for several last values of polynomial coefficients
    nframes = 10  # number of frames for frame_buffers to use
    average_left = []  # contains values to compare with coefficients of left_fit
    average_right = []  # contains values to compare with coefficients of right_fit
    margin = 1  # deviation margin between last frame and last {n}frames
    if len(frame_buffer_left) == 0:
        for i in range(nframes):
            frame_buffer_left.append(left_fit)
            frame_buffer_right.append(right_fit)
    else:
        for n in range(3):
            average_left.append(left_fit[n] / sum([i[n] for i in frame_buffer_left]) * nframes)
            average_right.append(left_fit[n] / sum([i[n] for i in frame_buffer_right]) * nframes)
    left = [1 for i in average_left if i < (1 - margin) or i > (1 + margin)]
    right = [1 for i in average_right if i < (1 - margin) or i > (1 + margin)]
    frame_buffer_left.append(left_fit)
    frame_buffer_right.append(right_fit)

    if len(left) == 0:
        frame_buffer_left.popleft()
        frame_buffer_left.append(left_fit)
    else:
        left_fit = frame_buffer_left[-1]

    if len(right) == 0:
        frame_buffer_right.popleft()
        frame_buffer_right.append(right_fit)
    else:
        right_fit = frame_buffer_right[-1]

    # we have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))
    # extracts left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # fits a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # generates x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # creates an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # generates a polygon to illustrate the search window area
    # and recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # draws the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result


# calculates curvature
def calc_curvature(path, files=0, sobel_kernel=3, mag_thresh=(35, 100), thresh_s=(170, 255)):
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    global frame_buffer_left  # collections.deque() for several last values of polynomial coefficients
    global frame_buffer_right  # collections.deque() for several last values of polynomial coefficients
    if files == 0:
        img = path
        left_fit, right_fit = finding_lines(img, sobel_kernel, mag_thresh, thresh_s, output=1)
        # left_fit, right_fit = frame_buffer_left[-1], frame_buffer_right[-1]
        y_eval = np.max(ploty)
        # For each y position generates random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2] for y in ploty])
        rightx = np.array([right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2] for y in ploty])
        # defines conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # fits new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # calculates the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                                 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                     1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # following lines of code calculate car center deviation from the middle point between to lanes
        # let's assume the car has 2m width
        y = 720
        leftx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
        rightx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
        lane_middle = (leftx + rightx) / 2
        center_deviation = abs((720 - lane_middle) / 720 * 2)
        return left_curverad, right_curverad, center_deviation
    else:
        n = len(files)
        for i in range(n):  # images calibrationXX, where XX = [2,4] are the most distorted from camera images
            # img = cv2.imread(path + files[i])
            img = mpimg.imread(path + files[i])
            frame_buffer_left = deque()
            frame_buffer_right = deque()
            left_fit, right_fit = finding_lines(img, sobel_kernel, mag_thresh, thresh_s, output=1)
            y_eval = np.max(ploty)
            # for each y position generates random x position within +/-50 pix
            # of the line base position in each case (x=200 for left, and x=900 for right)
            leftx = np.array([left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2] for y in ploty])
            rightx = np.array([right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2] for y in ploty])
            # defines conversions in x and y from pixels space to meters
            ym_per_pix = 30 / 720  # meters per pixel in y dimension
            xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
            # fits new polynomials to x,y in world space
            left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
            # calculates the new radii of curvature
            left_curverad = ((1 + (
                2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
            right_curverad = (
                                 (1 + (
                                     2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                         1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
            # now our radius of curvature is in meters
            # following lines of code calculate car center deviation from the middle point between to lanes
            # let's assume the car has 2m width
            y = 720
            leftx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
            rightx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
            lane_middle = (leftx + rightx) / 2
            center_deviation = abs((720 - lane_middle) / 720 * 2)
            print(left_curverad, 'm', right_curverad, 'm', center_deviation, 'm')


# draw text with a curvature on the image
def draw_text(img, text1, text2, text3):
    text4 = 'curvl:' + str(text1) + 'm ' + 'curvr:' + str(text2) + 'm'
    text5 = 'center:' + str(text3) + 'm'
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, text4, (200, 50), font, 1, (255, 255, 255), 2)
    img = cv2.putText(img, text5, (200, 80), font, 1, (255, 255, 255), 2)
    return img


# draws green zone on images where car can drive
def draw_lane_zone(img, output=0, src=np.float32([[200, 680], [590, 451], [690, 451], [1042, 666]]),
                   dst=np.float32([[180, 720], [200, 0], [1000, 0], [1000, 720]])):
    # Create an image to draw the lines on
    warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left_fit, right_fit = finding_lines(img, output=output)

    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]
    index = [n for n, (t, p) in enumerate(zip(left_fitx, right_fitx)) if t >= p]  # finds the point where left and right
    # polynomial intersect
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # draws the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # warps the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    left_curverad, right_curverad, center = calc_curvature(img)
    result = draw_text(result, left_curverad, right_curverad, center)
    return result


# process to apply to still images (*or single video frames)
def process_still_image(image):
    img = draw_lane_zone(image, output=2)
    return img


def process_video(videofilein, videofileout):
    # clears and initializes frame buffers
    global frame_buffer_left  # collections.deque() for several last values of polynomial coefficients
    global frame_buffer_right  # collections.deque() for several last values of polynomial coefficients
    global frame_curvature_left  # collections.deque() for several last values of curvature
    global frame_curvature_right  # collections.deque() for several last values of curvature
    frame_buffer_left = deque()
    frame_buffer_right = deque()
    frame_curvature_left = deque()
    frame_curvature_right = deque()
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,1)
    clip1 = VideoFileClip(videofilein)  # .subclip(20,30)
    write_clip = clip1.fl_image(process_still_image)  # NOTE: this function expects color images!!
    write_clip.write_videofile(videofileout, audio=False)


# gets random non vehicle image from path/subfolder/prefix{n}
def choose_random_non_vehicle(img, path='non-vehicle', subfolder='GTI', prefix='extra', n=1, ext='.png'):
    file = path + subfolder + prefix + str(n) + ext
    img = mpimg.imread(file)
    return img


import glob


# extracts pathes to images (cars and notcars)
def read_vehicles_not_vehicles(path_vehicle, path_not_vehicle):
    cars = []
    notcars = []
    images = glob.glob(path_vehicle + '/*/*.png')
    for image in images:
        cars.append(image)
    images = glob.glob(path_not_vehicle + '/*/*.png')
    for image in images:
        notcars.append(image)
    return cars, notcars


# function returns some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Defines a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Defines a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Reads in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Defines a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Defines a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Returns data_dict
    return data_dict


from skimage.feature import hog


# defines a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


# gets hog representation of the image
def get_hog_image(img, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features, hog_image = get_hog_features(gray, orient,
                                           pix_per_cell, cell_per_block,
                                           vis=vis, feature_vec=feature_vec)
    return hog_image


# Defines a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# Defines a function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Defines a function to compute binned color features
def spatial_hist_features(feature_image, spatial_size=(32, 32),
                          hist_bins=32, hist_range=(0, 256)):
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins)
    # Append the new feature vector to the features list
    return np.concatenate((spatial_features, hist_features))


# calculates hog features for one or all color channels
def hog_features(feature_image, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        # Compute individual channel HOG features for the entire image
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(feature_image[:, :, 0], orient, pix_per_cell, cell_per_block, vis=False,
                                feature_vec=False)
        hog2 = get_hog_features(feature_image[:, :, 1], orient, pix_per_cell, cell_per_block, vis=False,
                                feature_vec=False)
        hog3 = get_hog_features(feature_image[:, :, 2], orient, pix_per_cell, cell_per_block, vis=False,
                                feature_vec=False)
        hog_feature = np.concatenate((hog1, hog2, hog3))
    else:
        hog_feature = get_hog_features(feature_image[:, :, hog_channel], orient,
                                       pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    return hog_feature


# calculates features using spatial and color information and hog transformation
def spatial_color_hog_features(feature_image, spatial_size=(32, 32),
                               hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2,
                               hog_channel=0, \
                               spatial=True, hist=True, hog=True):
    spatial_features = ''
    hist_features = ''
    hog_feature = ''
    if spatial == True:
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
    if hist == True:
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins)
    if hog == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog1 = get_hog_features(feature_image[:, :, 0], orient, pix_per_cell, cell_per_block, vis=False,
                                    feature_vec=False)
            hog2 = get_hog_features(feature_image[:, :, 1], orient, pix_per_cell, cell_per_block, vis=False,
                                    feature_vec=False)
            hog3 = get_hog_features(feature_image[:, :, 2], orient, pix_per_cell, cell_per_block, vis=False,
                                    feature_vec=False)

            hog_feature = np.hstack((hog1, hog2, hog3))
            hog_feature = np.ravel(hog_feature)
        else:
            hog_feature = get_hog_features(feature_image[:, :, hog_channel], orient,
                                           pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    if spatial == hist == hog == True:
        result = np.concatenate((spatial_features, hist_features, hog_feature))
    elif spatial == hist == True and hog == False:
        result = np.concatenate((spatial_features, hist_features))
    elif spatial == hog == True and hist == False:
        result = np.concatenate((spatial_features, hog_feature))
    elif spatial == True and hist == hog == False:
        result = spatial_features
    elif spatial == False and hist == hog == True:
        result = np.concatenate((hist_features, hog_feature))
    elif spatial == hog == False and hist == True:
        result = hist_features
    elif spatial == hist == False and hog == True:
        result = hog_feature

    return result


# Defines a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, func, cspace='RGB', *args):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'RGB2YCrCb':
                feature_image = convert_color(image, conv='RGB2YCrCb')
        else:
            feature_image = np.copy(image)
        feature = func(image, *args)
        features.append(feature)
    # Return list of feature vectors
    return features


from sklearn.preprocessing import StandardScaler


# generates histograms for raw features and normalized features
def normalized_and_raw_features(cars, notcars, func, cspace='RGB', *args):
    car_features = extract_features(cars, func, cspace, *args)
    notcar_features = extract_features(notcars, func, cspace, *args)
    if len(car_features) > 0:
        # Creates an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fits a per-column scaler
        # Computes the mean and std to be used for later scaling.
        X_scaler = StandardScaler().fit(X)
        # Applies the scaler to X
        # Perform standardization by centering and scaling
        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(cars))
        # Plots an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        return fig
    else:
        print('function only returns empty feature vectors...')


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time


# trains classifier
def train_classifier(car_features, notcar_features, X_scaler, n_predict=0):
    # car_features = extract_features(cars, func, cspace, *args)
    # notcar_features = extract_features(notcars, func, cspace, *args)
    # Define a labels vector based on features lists
    y = np.hstack((np.ones(len(car_features)),
                   np.zeros(len(notcar_features))))
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    # Train the SVC
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    if n_predict > 0:
        # Check the prediction time for first {n} samples
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
    return svc


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window_simple(img, x_start_stop=(None, None), y_start_stop=(None, None),
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    x_start_stop = list(x_start_stop)
    y_start_stop = list(y_start_stop)
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


import math
import random


# Define a function that takes an image,
# and generates input data for function slide_window_simple():
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window_advanced(img, x_start_stop=(None, None), y_start_stop=(None, None),
                          xy_window=(64, 64), xy_overlap=(0.5, 0.5), rows=3):
    # idea behind sliding windows algorith to split the image into subareas from top to down in selected \
    # range x_start_stop=(None, None), y_start_stop=(None, None)
    # size of each area will be multiplies of height of smallest windows height xy_window[1]
    # since for us important vertical resolution of sliding windows due to perspective (more distant objects will be \
    # higher in image of flat road)
    # If x and/or y start/stop positions not defined, set to image size
    x_start_stop = list(x_start_stop)
    y_start_stop = list(y_start_stop)
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    n = math.floor(math.log(yspan // xy_window[1], 2))  # difference between sliding windows' sizes will be
    # will be power of 2, yspan = xy_window[1]*x^2
    windows = []
    y_start_stop_new = y_start_stop[1]  # range for sliding windows, to avoid to search with smaller windows \
    #  in front of a car where obviously sliding window should be bigger
    for i in range(0, n + 1):
        window_size = [x * 2 ** i for x in xy_window]
        row_tmp = rows
        while row_tmp != 0:
            if y_start_stop[0] + window_size[1] * (0.5 * (rows + 1)) <= y_start_stop[1]:
                y_start_stop_new = int(y_start_stop[0] + window_size[1] * (0.5 * (rows + 1)))
                row_tmp = 0
                break
            else:
                row_tmp -= 1
        windows.append(slide_window_simple(img, x_start_stop=x_start_stop, \
                                           y_start_stop=(y_start_stop[0], y_start_stop_new), \
                                           xy_window=(window_size[0], window_size[1]), xy_overlap=xy_overlap))
    return windows


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'RGB2YCrCb':
            feature_image = convert_color(img, conv='RGB2YCrCb')
    else:
        feature_image = np.copy(img)

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # list for image features (spatial, color histogram or hog)
        img_features = []
        # 3) Extract the test window from original image
        test_img = cv2.resize(feature_image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using spatial_color_hog_features, using spatial color information,
        # color histograms and hog transformation
        img_features = spatial_color_hog_features(test_img, spatial_size,
                                                  hist_bins, hist_range, orient, pix_per_cell, cell_per_block,
                                                  hog_channel, spatial_feat, hist_feat, hog_feat)

        # 5) Scale extracted features to be fed to classifier

        test_features = scaler.transform(np.array(img_features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# cuts out a part of the image
# base_size - smallest size of sliding window
# "scale" to scale sliding image new_window-size = base_size*scale x base_size*scale
# rows, how many of rows of sliding windows to consider
def image_crop(img, ystart):  # , base_size=64, scale=1, rows=3):
    img = np.copy(img)
    delta_y = 60  # offset of the region_of_interest from the middle of image
    delta_x = 10  # offset of the region_of_interest from the middle of image
    x_offset = 0  # offset of the region_of_interest's bottom corners (criticals for the challenge)
    y_offset = 0  # offset of the region_of_interest's bottom corners (critical for the challenge)
    x = img.shape[1]
    y = img.shape[0]
    vertices = np.array([[(x // 2 + 300, y - y_offset), (x // 2 - delta_x, ystart), \
                          (0, ystart), (0, y - y_offset)]], dtype=np.int32)
    # defines a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on the image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(img, vertices, ignore_mask_color)
    img_new = img

    return img_new


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, \
              spatial_feat=True, hist_feat=True, hog_feat=True):
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    window_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            if spatial_feat == hist_feat == hog_feat == True:
                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            elif spatial_feat == hist_feat == True and hog_feat == False:
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features)).reshape(1, -1))
            elif spatial_feat == True and hist_feat == hog_feat == False:
                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features)).reshape(1, -1))
            elif spatial_feat == hog_feat == True and hist_feat == False:
                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hog_features)).reshape(1, -1))
            elif spatial_feat == False and hist_feat == hog_feat == True:
                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((hist_features, hog_features)).reshape(1, -1))
            elif spatial_feat == hist_feat == False and hog_feat == True:
                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((hog_features)).reshape(1, -1))
            elif spatial_feat == hog_feat == False and hist_feat == True:
                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((hist_features)).reshape(1, -1))

            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                window_list.append(((xbox_left, ytop_draw + ystart), \
                                    (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return window_list


# function that generates list with sliding windows where cars were found
def find_cars_list(image, scale_min, scale_max, y_start_stop, svc, X_scaler, orient, \
                   pix_per_cell, cell_per_block, spatial_size, histbin, \
                   spatial_feat=True, hist_feat=True, hog_feat=True):
    window_list = []  # in which sliding windows cars located
    img_crop = image_crop(image, y_start_stop[0])  # , base_size=64, scale=1, rows=5)
    for scale in range(scale_min, 2 * scale_max, 1):
        # img_crop = image_crop(image, y_start_stop[0], base_size=64, scale=1, rows=5)
        window_list.append(find_cars(img_crop, y_start_stop[0], y_start_stop[1], scale, \
                                     svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, histbin, \
                                     spatial_feat, hist_feat, hog_feat))

    window_list = [i for c in window_list for i in c]
    return window_list


# creates heat map of detected images
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        # heatmaps with x coordinates < 700 will be ignored, since we know that we are in the right lane
        # using heatmaps together with LIDARs can help to narrow search field for cars in video stream
        if box[0][0] > 700 and box[1][0] > 700:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap


# apply threshold for heat map values
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


from scipy.ndimage.measurements import label


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


import pickle

heatmap_global = deque()  # buffer for heatmaps
frame_buffer = 2  # number or frames in buffer for heatmaps

spatial_feat = False
hist_feat = False
hog_feat = True


# process to apply to still images (*or single video frames)
def still_image_detect_cars(image):
    global image_number
    global heatmap_global
    global frame_buffer
    global spatial_feat
    global hist_feat
    global hog_feat
    scale_min = pickle.load(open('pickled_parameters/' + 'scale_min', 'rb'))
    scale_max = pickle.load(open('pickled_parameters/' + 'scale_max', 'rb'))
    y_start_stop = pickle.load(open('pickled_parameters/' + 'y_start_stop', 'rb'))
    svc = pickle.load(open('pickled_parameters/' + 'svc', 'rb'))
    X_scaler = pickle.load(open('pickled_parameters/' + 'X_scaler', 'rb'))
    orient = pickle.load(open('pickled_parameters/' + 'orient', 'rb'))
    pix_per_cell = pickle.load(open('pickled_parameters/' + 'pix_per_cell', 'rb'))
    cell_per_block = pickle.load(open('pickled_parameters/' + 'cell_per_block', 'rb'))
    spatial_size = pickle.load(open('pickled_parameters/' + 'spatial_size', 'rb'))
    histbin = pickle.load(open('pickled_parameters/' + 'histbin', 'rb'))
    window_list = find_cars_list(image, scale_min, scale_max, y_start_stop, svc, X_scaler, orient, \
                                 pix_per_cell, cell_per_block, spatial_size, histbin,
                                 spatial_feat, hist_feat, hog_feat)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, window_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    if len(heatmap_global) < frame_buffer:
        heatmap_global.append(heat)
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
    else:
        heatmap_global.append(heat)
        heat = heatmap_global.popleft()
        for i in range(len(heatmap_global)):
            heat = cv2.bitwise_and(heatmap_global[i], heat)  # help to filter out false recognition
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    img = draw_labeled_bboxes(image, labels)

    print('frame processed')
    return img


def video_detect_cars(videofilein, videofileout):
    global heatmap_global
    heatmap_global = deque()
    # clears and initializes frame buffers
    global frame_buffer_left  # collections.deque() for several last values of polynomial coefficients
    global frame_buffer_right  # collections.deque() for several last values of polynomial coefficients
    global frame_curvature_left  # collections.deque() for several last values of curvature
    global frame_curvature_right  # collections.deque() for several last values of curvature
    frame_buffer_left = deque()
    frame_buffer_right = deque()
    frame_curvature_left = deque()
    frame_curvature_right = deque()
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,1)
    clip1 = VideoFileClip(videofilein)  # .subclip(10,15)
    write_clip = clip1.fl_image(still_image_detect_cars)  # NOTE: this function expects color images!!
    write_clip.write_videofile(videofileout, audio=False)


def save_to_pickle(parameter, name):
    file = open("pickled_parameters/" + name, "wb")
    pickle.dump(parameter, file)
    file.close()


# main function with the pipeline
def main(argv):
    imageio.plugins.ffmpeg.download()
    if len(argv) == 5:
        path_calibration = str(argv[1]) + '/'
        path_vehicle = str(argv[2]) + '/'
        path_non_vehichle = str(argv[3]) + '/'
        path_test_images = str(argv[3]) + '/'

    else:
        path_calibration = 'camera_cal/'
        path_vehicle = 'vehicles/'
        path_not_vehicle = 'non-vehicles/'
        path_test_images = 'test_images'
    img = mpimg.imread('test_images' + '/test4.jpg')
    img = draw_lane_zone(img, output=1)
    import matplotlib
    matplotlib.image.imsave('output_images/' + 'ZoneBetweenLinesBig' + '.png', img)

    files = ['1.png']  # chooses random image from training samples
    fig1 = multiple_plot(path_vehicle + 'KITTI_extracted/', files, choose_random_non_vehicle, 'vehicle', 'non-vehicle', \
                         path_not_vehicle, 'GTI/', 'image', 10, '.png')
    fig1.savefig('output_images/' + 'car_not_car' + '.jpg')

    cars, notcars = read_vehicles_not_vehicles(path_vehicle, path_not_vehicle)  # reads pathes to vehicles/non-vehicles
    data_info = data_look(cars, notcars)

    print('there is a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:',
          data_info["data_type"])

    files = []  # files to display
    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars))
    img = cars[ind]
    files.append(img)
    ind = np.random.randint(0, len(notcars))
    img = notcars[ind]
    files.append(img)

    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    # displays hog represation of one vehicle and one notvehicle
    # since I believe that training images were made with already calibrated camera I won't apply any
    # calibration to this images
    fig2 = multiple_plot_uncalibrated('', files, get_hog_image, 'Original Image', 'Hog Visualisation', orient,
                                      pix_per_cell, cell_per_block, True, False)

    # performs under different binning scenarios
    spatial = 32
    spatial_size = (spatial, spatial)  # Spatial binning dimensions
    histbin = 32  # number of histogram color bins
    n_predict = 10
    bins_range = (0, 256)
    color_space = 'RGB2YCrCb'

    # generates histograms for raw features and normalized features
    fig3 = normalized_and_raw_features(cars, notcars, spatial_hist_features, color_space, (spatial, spatial), \
                                       histbin, bins_range)
    fig3.savefig('output_images/' + 'normalized_and_raw_features' + '.jpg')

    image = mpimg.imread(path_test_images + '/test1.jpg')

    # range for sliding windows where to search for cars in image
    x_start_stop = (None, None)
    y_start_stop = (image.shape[0] // 2, image.shape[0] - 40)
    xy_window = (64, 64)
    xy_overlap = (0.5, 0.5)
    rows = 5

    windows = slide_window_simple(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                                  xy_window=xy_window, xy_overlap=xy_overlap)
    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    mpimg.imsave('output_images/' + 'simple_sliding_windows' + '.png', window_img)

    windows = slide_window_advanced(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                                    xy_window=xy_window, xy_overlap=xy_overlap, rows=rows)
    window_img = draw_boxes(image, windows[0], color=(0, 0, 255), thick=6)
    for i in range(1, len(windows)):
        window = windows[i]
        delta1 = random.randint(100, 255)
        delta2 = random.randint(100, 255)
        delta3 = random.randint(100, 255)
        color = (delta1, delta2, delta3)
        window_img = draw_boxes(window_img, window, color=color, thick=6)
    window = [((0, y_start_stop[0]), (image.shape[1], y_start_stop[1]))]
    window_img = draw_boxes(window_img, window, color=(0, 255, 0), thick=10)
    mpimg.imsave('output_images/' + 'advanced_sliding_windows' + '.png', window_img)

    global spatial_feat  # flag to use or not spatial features
    global hist_feat  # flag to use or not color features
    global hog_feat  # flag to use or not hog features

    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    hog_channel = 'ALL'  # one or all channels use in hog features calculations

    # extract car features based on flags
    car_features = extract_features(cars, spatial_color_hog_features, color_space, spatial_size, \
                                    histbin, bins_range, orient, pix_per_cell, cell_per_block, \
                                    hog_channel, spatial_feat, hist_feat, hog_feat)
    # extract not car features based on flags
    notcar_features = extract_features(notcars, spatial_color_hog_features, color_space, spatial_size, \
                                       histbin, bins_range, orient, pix_per_cell, cell_per_block, \
                                       hog_channel, spatial_feat, hist_feat, hog_feat)
    # concatenate cars' and notcars' features
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Standardize features by removing the mean and scaling to unit variance
    #  	Compute the mean and std to be used for later scaling
    X_scaler = StandardScaler().fit(X)

    ## trains classifier based on flags for spatial, color and hog features  set before
    # svc = train_classifier(car_features, notcar_features, X_scaler, n_predict = n_predict)

    # folder with pictures of roads
    files = os.listdir(path_test_images)
    scale_min = 2
    scale_max = 8

    save_to_pickle(scale_min, 'scale_min')
    save_to_pickle(scale_max, 'scale_max')
    save_to_pickle(y_start_stop, 'y_start_stop')
    # save_to_pickle(svc,'svc')
    svc = pickle.load(open('pickled_parameters/' + 'svc', 'rb'))
    save_to_pickle(X_scaler, 'X_scaler')
    save_to_pickle(orient, 'orient')
    save_to_pickle(pix_per_cell, 'pix_per_cell')
    save_to_pickle(cell_per_block, 'cell_per_block')
    save_to_pickle(spatial_size, 'spatial_size')
    save_to_pickle(histbin, 'histbin')

    i = 0  # some index for the following cycle
    # check how classifier works on test images
    for file in files:
        print(path_test_images + '/' + file)

        image = mpimg.imread(path_test_images + '/' + file)
        window_list = find_cars_list(image, scale_min, scale_max, y_start_stop, svc, X_scaler, \
                                     orient, pix_per_cell, cell_per_block, spatial_size, histbin,
                                     spatial_feat, hist_feat, hog_feat)
        draw_img = np.copy(image)

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat, window_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(draw_img, labels)

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()

        fig.savefig('output_images/' + 'classifier_test_' + str(i) + '.jpg')
        i += 1

        # following steps apply the pipeline to video streams

        videofilein = 'project_video.mp4'
        videofileout = 'output_video/project_out.mp4'
        video_detect_cars(videofilein, videofileout)

        # # #
        # # videofilein = 'challenge_video.mp4'
        # # videofileout = 'output_video/challenge_video_out.mp4'
        # # process_video(videofilein, videofileout)
        # #
        # # videofilein = 'harder_challenge_video.mp4'
        # # videofileout = 'output_video/harder_challenge_video_out.mp4'
        # # process_video(videofilein, videofileout)


if __name__ == '__main__':
    main(sys.argv)
