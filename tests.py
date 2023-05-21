import cv2 as cv
import numpy as np
from constants import *
import time

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
    
    # frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        
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

            # Set the destination points for the real world.
            # In this case we are setting to project the image into a square of 480x480px
            destinationPoints = np.array([
                [0, 0],
                [DEFAULT_ASPECT_RATIO, 0],
                [DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO],
                [0, DEFAULT_ASPECT_RATIO]
            ])

            points = np.array(points).astype(np.int32)

            print("Points acquired: ", points)
            
            # Now we compute the Homography between the World and the Static Camera
            homograhy, mask = cv.findHomography(points, destinationPoints)

            # Break inner while since we get them and we computed the Homography
            break
        
    # Destroy the window used to retrieve the points
    cv.destroyWindow("Point detection")

    # Return the homography, even if not defined (None)
    return (None, None) if homograhy is None else (homograhy, mask)

def getInstrinsicMatrix(video):
    # Fattorizzare H, settando il centro dell'immagine come punto principale (cx, cy)
    # (fx, fy) uso la largezza dell'immagine della camera statica (provare anche a ruotare)
    # Alternativa, provare ad usare solvePnP()     
    # Get the video width and height, so we can get cx, cy, fx and fy
    _, frame = video.read()

    # frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

    width = int(frame.shape[1])
    height = int(frame.shape[0])
    
    cx = width // 2 # Get the integer value of cx
    cy = height // 2 # Get the integer value of cy
    
    fx = fy = height # Get the integer value of fx, fy
    
    # Now we can build K
    K  =  np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
    return K

def showCircleLightDirection(image):
    center_x = center_y = DEFAULT_ASPECT_RATIO // 2
    radius = DEFAULT_ASPECT_RATIO // 2

    cv.circle(image, (center_x, center_y), 2, (255, 255, 255), -1)
    
    cv.imshow("Circle", image)
    
image = np.zeros((DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO, 3), dtype=np.uint8)


# Load videos
video1 = cv.VideoCapture(STATIC_VIDEO_FILE_PATH)
video2 = cv.VideoCapture(MOVING_VIDEO_FILE_PATH)

# Initiate SIFT detector and FLANN Matcher for feature detection and matching between the two cameras
sift = cv.SIFT_create()
flann = cv.FlannBasedMatcher_create()

# Get the K for static camera
intrinsicStaticCamera = getInstrinsicMatrix(video1)

print(intrinsicStaticCamera)

# Get the homography of the static camera
homographyStaticCamera, maskStaticCamera = retrieveROI(video1)

if homographyStaticCamera is None:
    print("No homography calculated for static camera")
    exit(-1)

# Apply the shift to the videos so they are syched
video1.set(cv.CAP_PROP_POS_FRAMES, 33)
video2.set(cv.CAP_PROP_POS_FRAMES, 0)

previousTime = 0

while video1.isOpened() and video2.isOpened():
    
    # Get time elapsed between now and previous iteration
    time_elapsed = time.time() - previousTime
    
    # Get each frame of the video
    staticRet, staticFrame = video1.read()
    movingRet, movingFrame = video2.read()
    
    # staticFrame = cv.rotate(staticFrame, cv.ROTATE_90_CLOCKWISE)
    
    if staticRet != True or movingRet != True:
        break

    if time_elapsed > 1./DEFAULT_FPS_RATE:
        previousTime = time.time()
        
        # Convert to grayscale
        staticFrame = cv.cvtColor(staticFrame, cv.COLOR_BGR2GRAY)
        movingFrame = cv.cvtColor(movingFrame, cv.COLOR_BGR2GRAY)

        # Define world camera
        staticFrame = cv.warpPerspective(staticFrame, homographyStaticCamera, (DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO))

        # Now, let's try to compute the features of each pov
        kpStatic, desStatic = sift.detectAndCompute(staticFrame, None)
        kpMoving, desMoving = sift.detectAndCompute(movingFrame, None)

        # Perform feature matching using KNN (K-Nearest-Neighborhood) technique
        matches = flann.knnMatch(desStatic, desMoving, k=2)

        # Retrieve the good matches, to eliminate the outliers
        goodMatches = []
        for m1, m2 in matches:
            if m1.distance < 0.7 * m2.distance:
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

            # Now we compute the Homography between the Moving and Static Camera
            homographyStaticMoving, mask = cv.findHomography(srcPoints, dstPoints, cv.RANSAC, 5.0)

            matchesMask = mask.ravel().tolist()

            # Calculate R, T from homography
            # This homography is from World (W) to Moving Camera (C)
            R, T = findCameraExtrinsicsParameters(homographyStaticMoving, intrinsicStaticCamera)

            print("Homography\n", homographyStaticMoving)

            _, rotations, translations, _ = cv.decomposeHomographyMat(homographyStaticMoving, intrinsicStaticCamera)
            # R = rotations[0]
            # T = translations[0]
            print("Possible rotations with cv method\n", rotations[0])
            print("Possible translations with cv method\n", translations[0])

            # height, width = staticFrame.shape
            # destinationPoints = np.float32([
            #     [0, 0],
            #     [0, height - 1],
            #     [width - 1, height - 1],
            #     [width - 1, 0]
            # ]).reshape(-1, 1, 2)
            
            # perspectiveTransformation = cv.perspectiveTransform(destinationPoints, H)
            # movingFrame = cv.polylines(movingFrame, [np.int32(perspectiveTransformation)], True, 255, 3, cv.LINE_AA)
        else:
            R = T = []
            matchesMask = None

        drawParams = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matchesMask,
            flags=2
        )

        # If both rotation matrix and translation vector are defined, then we can estimate the light direction
        if len(R) != 0 and len(T) != 0:
            print("Rotation matrix\n", R)
            print("Translation vector\n", T)
            R = R.T
            print("Rotation matrix transpose\n", R)
            R = -1 * R
            print("Rotation matrix negated\n", R)
            l = np.dot(R, T)
            print("Vector light not normalised\n", l)
            norm_l = np.linalg.norm(l)
            print("Norm of the light vector\n", norm_l)
            ligth_vector = l / norm_l
            print("Vector value\n", ligth_vector)
            cv.imshow("World frame", staticFrame)
            showCircleLightDirection()
            break
            

        # cv.putText(staticFrame, 'Frame ' + str(video1.get(cv.CAP_PROP_POS_FRAMES)) + " of " +
        #         str(video1.get(cv.CAP_PROP_FRAME_COUNT)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)

        # cv.putText(movingFrame, 'Frame ' + str(video2.get(cv.CAP_PROP_POS_FRAMES)) + " of " +
        #         str(video2.get(cv.CAP_PROP_FRAME_COUNT)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)

        # siftMatches = cv.drawMatches( staticFrame, kpStatic, movingFrame, kpMoving, goodMatches, None, **drawParams)
        # cv.imshow("Matches", siftMatches)

        # cv.imshow("Frame 1", staticFrame)
        # cv.imshow("Frame 2", movingFrame)

    # Press Q on the keyboard to exit.
    if (cv.waitKey(25) & 0xFF == ord('q')):
        break
    

# Release videos
video1.release()
video2.release()

# And destroy windows
# cv.destroyAllWindows()
