import cv2 as cv
import numpy as np

from constants import *

def getPointFromImage(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        params[0].append(np.array((x, y), dtype=np.float32))
        print("Point added")
    pass

# H -> Homography
# K -> Intrinsic Parameters of the camera
# The function computes the extrinsic parameters using the homography and intrinsic parameters array.
# At the end, it returns the derived rotation and translation vectors
def findCameraExtrinsicsParameters(H, K):
    # Get the transpose of the Homography
    H = H.T
    # Get the inverse of the intrisinc parameters
    K_inverse = np.linalg.inv(K)
    # First column of the Homography
    h1 = H[0]
    h2 = H[1]                                           # Second column
    h3 = H[2]                                           # Third column
    alpha = 1 / np.linalg.norm(np.dot(K_inverse, h1))   # Scale factor
    r1 = alpha * np.dot(K_inverse, h1)  #  Rotation matrix first column
    r2 = alpha * np.dot(K_inverse, h2)  #  Rotation matrix second column
    r3 = np.cross(r1, r2)  #  Rotation matrix third column
    # Get the translation vector
    T = alpha * (K_inverse @ h3.reshape(3, 1))
    R = np.array([[r1], [r2], [r3]])  #  Get the rotation matrix
    R = np.reshape(R, (3, 3))
    return R, T                                         # Return extrinsic parameters

def retrieveROI(video):
    # Set video to initial frame, in order to get it
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    # ... and get the first frame
    _, frame = video.read()
    
    # Array used to store points, to then use them to calculate the homography
    points = []

    # 3x3 Matrix which will contain the homography between static camera and world
    homograhy = None

    # Now, from the first frame, let's try to compute Homography between World and the Static Camera
    while True:
        
        cv.imshow("Point detection", frame)
        cv.setMouseCallback("Point detection", getPointFromImage, param=[points])

        # Press Q on the keyboard to exit.
        if (cv.waitKey(25) & 0xFF == ord('q')):
            break

        for i in range(len(points)):
            cv.line( frame, tuple(points[i].astype(int)), tuple(points[(i + 1) % len(points)].astype(int)), (0, 0, 255), 3)

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
            homograhy, _ = cv.findHomography(np.array(points), destinationPoints)

            # Break inner while since we get them and we computed the Homography
            break
        
    # Destroy the window used to retrieve the points
    cv.destroyWindow("Point detection")

    # Return the homography, even if not defined (None)
    return None if homograhy is None else homograhy

# Load videos
video1 = cv.VideoCapture(STATIC_VIDEO_FILE_PATH)
video2 = cv.VideoCapture(MOVING_VIDEO_FILE_PATH)

print("Video 1 FPS: ", video1.get(cv.CAP_PROP_FPS))
print("Video 2 FPS: ", video2.get(cv.CAP_PROP_FPS))

# Set same frame rate
video1.set(cv.CAP_PROP_FPS, DEFAULT_FPS_RATE)
video2.set(cv.CAP_PROP_FPS, DEFAULT_FPS_RATE)

print("Video 1 FPS: ", video1.get(cv.CAP_PROP_FPS))
print("Video 2 FPS: ", video2.get(cv.CAP_PROP_FPS))

# Initiate SIFT detector and FLANN Matcher for feature detection and matching between the two cameras
sift = cv.SIFT_create()
flann = cv.FlannBasedMatcher_create()

# Get the homography of the static camera
homographyStaticCamera = retrieveROI(video1)

if homographyStaticCamera is None:
    print("No homography calculated for static camera")
    exit(-1)

# Apply the shift to the videos so they are syched
video1.set(cv.CAP_PROP_POS_FRAMES, 33)
video2.set(cv.CAP_PROP_POS_FRAMES, 0)

while video1.isOpened() and video2.isOpened():
    # Get each frame of the video
    staticRet, staticFrame = video1.read()
    movingRet, movingFrame = video2.read()

    if staticRet != True or movingRet != True:
        break

    # Convert to grayscale
    staticFrame = cv.cvtColor(staticFrame, cv.COLOR_BGR2GRAY)
    movingFrame = cv.cvtColor(movingFrame, cv.COLOR_BGR2GRAY)

    # Define world camera
    staticFrame = cv.warpPerspective(
        staticFrame, homographyStaticCamera, (DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO))

    # SIFT MATCH BETWEEN ROI AND MOVING CAMERA
    # Now, let's try to compute the features of each pov
    kpStatic, desStatic = sift.detectAndCompute(staticFrame, None)
    kpMoving, desMoving = sift.detectAndCompute(movingFrame, None)

    matches = flann.knnMatch(desStatic, desMoving, k=2)

    goodMatches = []
    for m1, m2 in matches:
        if m1.distance < 0.75 * m2.distance:
            goodMatches.append(m1)

    print("Matched points", len(goodMatches))

    # Here at this point, after computing the good matches we can try to compute the homography H21 (Moving with Static)
    if len(goodMatches) > MIN_MATCH_COUNT:
        srcPoints = np.float32([
            kpStatic[m.queryIdx].pt for m in goodMatches
        ]).reshape(-1, 1, 2)
        dstPoints = np.float32([
            kpMoving[m.trainIdx].pt for m in goodMatches
        ]).reshape(-1, 1, 2)

        # # Now we compute the Homography between the Moving and Static Camera
        homographyStaticMoving, mask = cv.findHomography(
            srcPoints, dstPoints, cv.RANSAC, 5.0)

        matchesMask = mask.ravel().tolist()

        height, width = staticFrame.shape
        destinationPoints = np.float32([
            [0, 0],
            [0, height - 1],
            [width - 1, height - 1],
            [width - 1, 0]
        ]).reshape(-1, 1, 2)

        perspectiveTransformation = cv.perspectiveTransform(
            destinationPoints, homographyStaticMoving)

        movingFrame = cv.polylines(movingFrame, [np.int32(
            perspectiveTransformation)], True, 255, 3, cv.LINE_AA)
    else:
        matchesMask = None

    drawParams = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=2
    )

    cv.putText(staticFrame, 'Frame ' + str(video1.get(cv.CAP_PROP_POS_FRAMES)) + " of " +
               str(video1.get(cv.CAP_PROP_FRAME_COUNT)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)

    cv.putText(movingFrame, 'Frame ' + str(video2.get(cv.CAP_PROP_POS_FRAMES)) + " of " +
               str(video2.get(cv.CAP_PROP_FRAME_COUNT)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)

    siftMatches = cv.drawMatches(
        staticFrame, kpStatic, movingFrame, kpMoving, goodMatches, None, **drawParams)

    cv.imshow("Matches", siftMatches)

    # H = homographyStaticCamera @ homographyStaticMoving
    # if homographyStaticMoving is not None:
    #     movingFrame = cv.warpPerspective(
    #         movingFrame, homographyStaticMoving, (movingFrame.shape[1], movingFrame.shape[0]))

    # cv.imshow("Moving frame",movingFrame)

    # world = cv.warpPerspective(movingFrame, H, (DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO))

    # cv.imshow("Homography frames", world)

    # siftMatches = cv.drawMatches(staticFrame, kpStatic, movingFrame, kpMoving, goodMatches, None, flags=2)

    # cv.imshow("SIFT Matches", siftMatches)

    # Press Q on the keyboard to exit.
    if (cv.waitKey(25) & 0xFF == ord('q')):
        break

# Release videos
video1.release()
video2.release()

# And destroy windows
cv.destroyAllWindows()
