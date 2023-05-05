import cv2 as cv
import numpy as np

from constants import *

# Load the two videos and extract their audio
# Frame difference: 82 First video start sooner
video1 = cv.VideoCapture(STATIC_VIDEO_FILE_PATH)
video2 = cv.VideoCapture(MOVING_VIDEO_FILE_PATH)

video1.set(cv.CAP_PROP_FPS, 30)
video2.set(cv.CAP_PROP_FPS, 30)

video1.set(cv.CAP_PROP_POS_FRAMES, 33)
video1.set(cv.CAP_PROP_POS_FRAMES, 0)

points = []

# TEST WITH FIDUAL MARKER FIND VIA MOUSE POINTS
hasFoundPoints = False
hasDestroyedPointsWindow = False

# Initiate SIFT detector
sift = cv.SIFT_create()
# bf = cv.BFMatcher()

# Initialize FLANN-based matcher
flann = cv.FlannBasedMatcher_create()

def getPointFromImage(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append(np.array((x, y), dtype=np.float32))
        print("Point added")
    pass

    
def applyUndistortion(frame, mtx, dist):
        
    # Get undistortion
    undistFrame = cv.undistort(frame, mtx, dist)
        
    # Lastly, return the undistorted frame
    return undistFrame

# H -> Homography
# K -> Intrinsic Parameters of the camera
# The function computes the extrinsic parameters using the homography and intrinsic parameters array.
# At the end, it returns the derived rotation and translation vectors
def findCameraExtrinsicsParameters(H, K):
    H = H.T                                             # Get the transpose of the Homography
    K_inverse = np.linalg.inv(K)                        # Get the inverse of the intrisinc parameters
    h1 = H[0]                                           # First column of the Homography
    h2 = H[1]                                           # Second column
    h3 = H[2]                                           # Third column
    alpha = 1 / np.linalg.norm(np.dot(K_inverse, h1))   # Scale factor
    r1 = alpha * np.dot(K_inverse, h1)                  #  Rotation matrix first column
    r2 = alpha * np.dot(K_inverse, h2)                  #  Rotation matrix second column
    r3 = np.cross(r1, r2)                               #  Rotation matrix third column
    T = alpha * (K_inverse @ h3.reshape(3, 1))          # Get the translation vector
    R = np.array([[r1], [r2], [r3]])                    #  Get the rotation matrix
    R = np.reshape(R, (3, 3))
    return R, T                                         # Return extrinsic parameters

# Get the distortion params from the file (for moving camera)
distVideoMoving = np.loadtxt(MOVING_VIDEO_PARAMS_PATH + "distortionCoeffs.dat")
mtxVideoMoving = np.loadtxt(MOVING_VIDEO_PARAMS_PATH + "intrinsicMatrix.dat")

distVideoStatic = np.loadtxt(STATIC_VIDEO_PARAMS_PATH + "distortionCoeffs.dat")
mtxVideoStatic = np.loadtxt(STATIC_VIDEO_PARAMS_PATH + "intrinsicMatrix.dat")

while video1.isOpened() and video2.isOpened():
    # Get each frame of the video
    staticRet, staticFrame = video1.read()
    movingRet, movingFrame = video2.read()

    if staticRet != True or movingRet != True:
        break

    identityFrame = np.array(staticFrame)

    # Now, from the first frame, let's try to compute Homography between World and the Static Camera
    while hasFoundPoints == False:

        cv.imshow("Point detection", identityFrame)
        cv.setMouseCallback("Point detection", getPointFromImage)

        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break

        for i in range(len(points)):
            cv.line(identityFrame, tuple(points[i].astype(int)), tuple(points[(i + 1) % len(points)].astype(int)), (0, 0, 255), 3)

        if len(points) == 4:

            print("Points acquired: ", points)

            # Set the destination points for the real world.
            # In this case we are setting to project the image into a square of 480x480px
            destinationPoints = np.array([
                [0, 0],
                [DEFAULT_ASPECT_RATIO, 0],
                [DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO],
                [0, DEFAULT_ASPECT_RATIO]
            ])

            # Now we compute the Homography between the World and the Static Camera
            homographyStaticCamera, _ = cv.findHomography(np.array(points), destinationPoints)
            
            # rvecStatic, tvecStatic = findCameraExtrinsicsParameters(homographyStaticCamera, mtxVideoStatic)
            
            # print(rvecStatic, tvecStatic)

            # Set that the key points has been found and stored
            hasFoundPoints = True

            # Break inner while since we get them and we computed the Homography
            break

    if len(points) != 4:
        print("Not enough points")
        break

    # Boolean check to remove the first window, used to obtain the points
    if hasDestroyedPointsWindow == False:
        cv.destroyWindow("Point detection")
        hasDestroyedPointsWindow = True

    # staticFrame = applyUndistortion(staticFrame, mtxVideoStatic, distVideoStatic)
    # movingFrame = applyUndistortion(movingFrame, mtxVideoMoving, distVideoMoving)

    # Define world camera
    staticFrame = cv.warpPerspective(staticFrame, homographyStaticCamera, (DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO))

    # SIFT MATCH BETWEEN ROI AND MOVING CAMERA
    # Now, let's try to compute the features of each pov
    kpStatic, desStatic = sift.detectAndCompute(staticFrame, None)
    kpMoving, desMoving = sift.detectAndCompute(movingFrame, None)

    matches = flann.knnMatch(desStatic, desMoving, k=2)

    goodMatches = []
    for m1, m2 in matches:
        if m1.distance < 0.7 * m2.distance:
            goodMatches.append(m1)

    print("Matched points", len(goodMatches))
    
    # Here at this point, after computing the good matches we can try to compute the homography H21 (Moving with Static)
    if len(goodMatches) > 50:

        srcPoints = np.float32([kpStatic[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPoints = np.float32([kpMoving[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

        # # Now we compute the Homography between the Moving and Static Camera
        homographyStaticMoving, _ = cv.findHomography(srcPoints, dstPoints, cv.RANSAC, 5.0)
        # if homographyStaticMoving is not None:
        #     H = homographyStaticCamera @ homographyStaticMoving
        
        # movingFrame = cv.warpPerspective(movingFrame, homographyStaticMoving, (movingFrame.shape[1], movingFrame.shape[0]))    
    
    # cv.imshow("Moving frame",movingFrame)
    
    # world = cv.warpPerspective(movingFrame, H, (DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO))
    
    # cv.imshow("Homography frames", world)

    siftMatches = cv.drawMatches(staticFrame, kpStatic, movingFrame, kpMoving, goodMatches, None, flags=2)

    cv.imshow("SIFT Matches", siftMatches)

    # Press Q on the keyboard to exit.
    if (cv.waitKey(25) & 0xFF == ord('q')):
        break

# Release videos
video1.release()
video2.release()

# And destroy windows
cv.destroyAllWindows()