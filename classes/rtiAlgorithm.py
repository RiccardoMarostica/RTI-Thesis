import cv2 as cv
import numpy as np
import os
from constants import *

from classes.video import Video

class LightDirection:
    def __init__(self, frame, lightVector) -> None:
        self.frame = frame
        self.ligthVector = lightVector
        pass

class RTI:
    def __init__(self) -> None:
        # Create methods to perform feature matching
        self.sift = cv.SIFT_create()
        self.flann = cv.FlannBasedMatcher_create()
        
        self.lightDirections = []
        pass

    def getDefaultK(self, video : Video):
        # Fattorizzare H, settando il centro dell'immagine come punto principale (cx, cy)
        # (fx, fy) uso la largezza dell'immagine della camera statica (provare anche a ruotare)
        # Alternativa, provare ad usare solvePnP()
        # Get the video width and height, so we can get cx, cy, fx and fy
        _, frame = video.getCurrentFrame()

        # frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

        width = int(frame.shape[1])
        height = int(frame.shape[0])

        cx = width // 2  # Get the integer value of cx
        cy = height // 2  #  Get the integer value of cy

        fx = fy = height  # Get the integer value of fx, fy

        # Now we can build K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
        return K

    def getPointFromImage(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            params[0].append(np.array((x, y), dtype=np.float32))
            print("Point added")
        pass

    def getWorldHomography(self, video: Video):
        
        # Set video to initial frame, in order to get it
        video.setVideoFrame()

        # ... and get the first frame
        _, frame = video.getCurrentFrame()

        # Array used to store points, to then use them to calculate the homography
        points = []

        # 3x3 Matrix which will contain the homography between static camera and world
        homograhy = None

        # Now, from the first frame, let's try to compute Homography between World and the Static Camera
        while True:
            # Show the frame and add a mouse callback to get the 4 points
            cv.imshow("Point detection", frame)
            cv.setMouseCallback("Point detection", self.getPointFromImage, param=[points])

            # Press Q on the keyboard to exit.
            if (cv.waitKey(25) & 0xFF == ord('q')):
                break

            # For simplicity, draw lines between points
            for i in range(len(points)):
                cv.line(frame, tuple(points[i].astype(int)), tuple(points[(i + 1) % len(points)].astype(int)), (0, 0, 255), 3)

            if len(points) == 4:

                # Set the destination points for the real world.
                # In this case we are setting to project the image into a square of 480x480px
                destinationPoints = np.array([
                    [0, 0],
                    [DEFAULT_SQUARE_SIZE, 0],
                    [DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE],
                    [0, DEFAULT_SQUARE_SIZE]
                ])

                points = np.array(points).astype(np.int32)

                # Now we compute the Homography between the World and the Static Camera
                homograhy, _ = cv.findHomography(points, destinationPoints)

                # Break inner while since we get them and we computed the Homography
                break

        # Destroy the window used to retrieve the points
        cv.destroyWindow("Point detection")
        cv.destroyAllWindows()

        # Return the homography, even if not defined (None)
        return [] if homograhy is None else homograhy

    '''
    H -> Homography
    K -> Intrinsic Parameters of the camera
    The function computes the extrinsic parameters using the homography and intrinsic parameters array.
    At the end, it returns the derived rotation and translation vectors
    '''
    def getExtrinsicsParameters(self, H, K):
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
        return R, T

    def getHomographyWithFeatureMatching(self, frame1, frame2):
        # Compute features using SIFT for both frames
        keypoints1, descriptors1 = self.sift.detectAndCompute(frame1, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(frame2, None)

        # Perform feature matching using KNN (K-Nearest-Neighborhood) technique
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)

        # ... and remove outliers
        goodMatches = []
        for m1, m2 in matches:
            if m1.distance < 0.7 * m2.distance:
                goodMatches.append(m1)

        if len(goodMatches) > MIN_MATCH_COUNT:
            # Get source and destination points found inside the good matches to build the homography between the two frames
            src = np.float32([keypoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            dst = np.float32([keypoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            
            # Build the homography
            homography, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)            
            return homography
        else:
            return []
        
    def getLightVector(self, R, T):
        R = R.T
        R = -1 * R
        l = np.dot(R, T)
        norm_l = np.linalg.norm(l)
        lightVector = l / norm_l
        return lightVector
        
    def storeLightVector(self, frame, lightVector):
        # store light direction in a specific array
        self.lightDirections.append(LightDirection(frame, lightVector))
            
    def showCircleLightDirection(self, light_direction):    
        # Create a blank image
        image = np.zeros((DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO, 3), dtype=np.uint8)

        center_x = center_y = DEFAULT_ASPECT_RATIO // 2
        radius = DEFAULT_ASPECT_RATIO // 2    
            
        # Draw the circle border
        cv.circle(image, (center_x, center_y), radius, (255, 255, 255), 1)
        cv.line(image, (0, center_y), (DEFAULT_ASPECT_RATIO, center_y), (255, 255, 255), 1)
        cv.line(image, (center_x, 0), (center_x, DEFAULT_ASPECT_RATIO), (255, 255, 255), 1)
        
        if len(light_direction) != 0:
            x = int(((light_direction[0][0] + 1) * DEFAULT_ASPECT_RATIO) / 2)
            y = int(((light_direction[1][0] + 1) * DEFAULT_ASPECT_RATIO) / 2)        
            cv.putText(image, "P = (" + str(x) + ", " + str(y) + ")", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv.line(image, (center_x, center_y), (int(x), int(y)), (0, 255, 0), 1)
        
        return image
