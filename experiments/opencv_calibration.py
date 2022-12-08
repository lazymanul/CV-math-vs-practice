import numpy as np
import cv2 as cv
import glob

img_path_expr = '../assets/gen_dataset/*.png'

# checkerboard paprameters
x_points = 13
y_points = 9

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((y_points*x_points,3), np.float32)
objp[:,:2] = np.mgrid[0:x_points,0:y_points].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(img_path_expr)

print(images)
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('img', gray)
    cv.waitKey(2000)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (x_points, y_points), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (x_points, y_points), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(2500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.set_printoptions(precision=6, suppress=True)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)