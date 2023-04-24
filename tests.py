import cv2 as cv
import numpy as np

from constants import *

# Load the two videos and extract their audio
# Frame difference: 82 First video start sooner
video1 = cv.VideoCapture(STATIC_VIDEO_FILE_PATH)
video2 = cv.VideoCapture(MOVING_VIDEO_FILE_PATH)

video1.set(cv.CAP_PROP_FPS, 30)
video2.set(cv.CAP_PROP_FPS, 30)

video1.set(cv.CAP_PROP_POS_FRAMES, 82)

points = []

# TEST WITH FIDUAL MARKER FIND VIA MOUSE POINTS
hasFoundPoints = False
hasDestroyedPointsWindow = False

# Initiate SIFT detector
sift = cv.SIFT_create()
bf = cv.BFMatcher()

# Initialize FLANN-based matcher
flann = cv.FlannBasedMatcher_create()


def getPointFromImage(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append(np.array((x, y), dtype=np.float32))
        print("Point added")
    pass

# H -> Homography
# K -> Intrinsic Parameters of the camera
# The function computes the extrinsic parameters using the homography and intrinsic parameters array.
# At the end, it returns the derived rotation and translation vectors


def findCameraExtrinsicsParameters(H, K):
    H = H.T                                         # Get the transpose of the Homography
    K_inverse = np.linalg.inv(K)                    # Get the inverse of the intrisinc parameters
    h1 = H[0]                                       # First column of the Homography
    h2 = H[1]                                       # Second column
    h3 = H[2]                                       # Third column
    L = 1 / np.linalg.norm(np.dot(K_inverse, h1))   # Scale factor
    r1 = L * np.dot(K_inverse, h1)                  #  Rotation matrix first column
    r2 = L * np.dot(K_inverse, h2)                  #  Rotation matrix second column
    r3 = np.cross(r1, r2)                           #  Rotation matrix third column
    T = L * (K_inverse @ h3.reshape(3, 1))          # Get the translation vector
    R = np.array([[r1], [r2], [r3]])                #  Get the rotation matrix
    R = np.reshape(R, (3, 3))
    return R, T                                     # Return extrinsic parameters


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

    # Define world camera
    world = cv.warpPerspective(staticFrame, homographyStaticCamera, (DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO))

    # Get the intrinsic parameters from static video
    mtxVideoStatic = np.loadtxt(STATIC_VIDEO_PARAMS_PATH + "intrinsicMatrix.dat")
    distVideoStatic = np.loadtxt(STATIC_VIDEO_PARAMS_PATH + "distortionCoeffs.dat")

    # ... and retrieve extrinsic parameters
    rvec, tvec = findCameraExtrinsicsParameters(homographyStaticCamera, mtxVideoStatic)
    
    u, s, v = np.linalg.svd(rvec)
            
    print(u, s, v)
    
    break

    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtxVideoStatic, distVideoStatic)

    # SIFT MATCH BETWEEN ROI AND MOVING CAMERA
    # Now, let's try to compute the features of each pov
    kpStatic, desStatic = sift.detectAndCompute(world, None)
    kpMoving, desMoving = sift.detectAndCompute(movingFrame, None)

    # matches = bf.knnMatch(desROI, desMoving, k = 2)
    matches = flann.knnMatch(desStatic, desMoving, k=2)

    goodMatches = []
    for m1, m2 in matches:
        if m1.distance < 0.7 * m2.distance:
            goodMatches.append(m1)

    print("Matched points", len(goodMatches))

    # Here at this point, after computing the good matches we can try to compute the homography H21 (Moving with Static)
    if len(goodMatches) > 10:

        srcPoints = np.float32([kpStatic[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPoints = np.float32([kpStatic[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

        # # Now we compute the Homography between the Moving and Static Camera
        homographyStaticMoving, _ = cv.findHomography(srcPoints, dstPoints)

        if homographyStaticMoving is not None:
            # Get the distortion params from the file (for moving camera)
            distVideoMoving = np.loadtxt(MOVING_VIDEO_PARAMS_PATH + "distortionCoeffs.dat")
            mtxVideoMoving = np.loadtxt(MOVING_VIDEO_PARAMS_PATH + "intrinsicMatrix.dat")
            
            # ... and retrieve extrinsic parameters
            rvec, tvec = findCameraExtrinsicsParameters(homographyStaticMoving, mtxVideoMoving)
            
            axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
            imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtxVideoMoving, distVideoMoving)
            
            # img = cv.line(movingFrame, (0,0), tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
            # img = cv.line(movingFrame, (0,0), tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
            # img = cv.line(movingFrame, (0,0), tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)

    siftMatches = cv.drawMatches(world, kpStatic, movingFrame, kpMoving, goodMatches, None, flags=2)

    cv.imshow("SIFT Matches", siftMatches)

    # Press Q on the keyboard to exit.
    if (cv.waitKey(25) & 0xFF == ord('q')):
        break

# Release videos
video1.release()

# And destroy windows
cv.destroyAllWindows()


# TEST WITH FIDUAL MARKER TO FIND
# # While one of the two videos is open, then read frame by frame
# while video1.isOpened():

#     # Get the frame from each video
#     ret1, frame1 = video1.read()

#     # If one of the two does not return a frame, then exit the loop
#     if not ret1:
#         break

#     # Convert the frame in grayscale
#     grayFrame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

#     # Now, let's apply the threshold to the frame
#     _, threshold = cv.threshold(grayFrame, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

#     # Then, let's find the contours of the black square marker
#     contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#     contours = sorted(contours, key = cv.contourArea)

#     squareContour = contours[-2]

#     # # squareContour = max(contours, key = cv.contourArea)

#     epsilon = 0.1 * cv.arcLength(squareContour, True)
#     approxCorners = cv.approxPolyDP(squareContour, epsilon, True)

#     corners = approxCorners.reshape(-1, 2)

#     frame1 = cv.rectangle(frame1, corners[0], corners[2], (0, 255, 0), 3)

#     distances = [cv.norm(corners[i], corners[i+1]) for i in range(len(corners)-1)]

#     minDistance = int(np.min(distances))
#     # Find the smallest value for the first position of the nested arrays
#     x = np.amin(corners, axis=0)[0]

#     # Find the smallest value for the second position of the nested arrays
#     y = np.amin(corners, axis=0)[1]

#     # x, y =corners[0]

#     # Define ROI using coordinates of smallest circle and size of internal square
#     roi_x, roi_y, roi_w, roi_h = x, y, minDistance, minDistance

#     roi = frame1[roi_y : (roi_y + roi_h), roi_x : (roi_x + roi_w)]

#     cv.putText(roi, "X, Y: " + str(x) + ", " + str(y), (50, 50), fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 0), thickness = 3)

#     # cv.imshow("FRAME", roi)

#     mask = np.zeros_like(grayFrame)
#     cv.fillPoly(mask, [corners], (255, 255, 255))

#     minDistanceInner = minDistance

#     dstPoints = np.array([ [0, 0], [0, minDistanceInner], [minDistanceInner, minDistanceInner], [minDistanceInner, 0]])

#     homography = cv.findHomography(corners, dstPoints, cv.RANSAC)

#     worldImage = cv.warpPerspective(frame1, homography[0], (minDistanceInner, minDistanceInner))

#     cv.imshow("World", worldImage)

#     # distances = []
#     # for i in range(len(approxCorners)):

#     #     j = (i + 1) % len(approxCorners)
#     #     print(approxCorners[i][0], approxCorners[j][0])
#     #     frame1 = cv.line(frame1, (approxCorners[i][0][0], approxCorners[i][0][1]), (approxCorners[j][0][0], approxCorners[j][0][1]), (0, 255, 0), 3)

#     #     distance = cv.norm(approxCorners[i][0], approxCorners[j][0])
#     #     distances.append(distance)


#     # cv.imshow("Frame", roi)

#     # minDistanceIndex = distances.index(min(distances))
#     # x, y = approxCorners[minDistanceIndex][0]

#     # frame1 = cv.circle(frame1, (x,y), 1, (0, 255, 0), 3)

#     # # Define ROI using coordinates of smallest circle and size of internal square
#     # roi_x, roi_y, roi_w, roi_h = x, y, 670, 673

#     # roi = frame1[roi_y : (roi_y + roi_h), (roi_x - roi_w) :roi_x]

#     # Find the circles in the corners of the square
#     # corners = approxCorners.reshape(-1, 2)
#     # distances = [cv.norm(corners[i], corners[i+1]) for i in range(len(corners)-1)]
#     # min_distance_index = distances.index(min(distances))

#     # # Find the smallest circle in the left corner of the square
#     # (x, y), radius = cv.minEnclosingCircle(approxCorners[min_distance_index])
#     # circle = (int(x), int(y))

#     # Crop the frame to only include the ROI
#     # roi = frame1[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

#     # cv.imshow('ROI', frame1)

#     mask = np.zeros_like(grayFrame)
#     cv.fillPoly(mask, [approxCorners], (255, 255, 255))

#     minDistanceInner = int(minDistance / 1.4)

#     dstPoints = np.array([ [0, 0], [0, minDistanceInner], [minDistanceInner, minDistanceInner], [minDistanceInner, 0]])

#     homography = cv.findHomography(approxCorners, dstPoints, cv.RANSAC)

#     worldImage = cv.warpPerspective(frame1, homography[0], (minDistanceInner, minDistanceInner))

#     # grayWorldImage = cv.cvtColor(worldImage, cv.COLOR_BGR2GRAY)

#     # sift = cv.SIFT_create()
#     # keypoints = sift.detect(grayWorldImage, None)

#     # worldImage = cv.drawKeypoints(grayWorldImage, keypoints, worldImage)

#     # cv.imshow("Frame", worldImage)

#     # Press Q on the keyboard to exit.
#     if (cv.waitKey(25) & 0xFF == ord('q')):
#         break

# # Release videos
# video1.release()

# # And destroy windows
# cv.destroyAllWindows()
