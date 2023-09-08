import cv2 as cv
import numpy as np
from constants import *

from classes.video import Video
from classes.parameters import Parameters

class VideoAnalysis:
    def __init__(self) -> None:
        """The constructor inistalise SIFT to get features from a frame, and FLANN for feature matching.
        Also, it initialise a list used to store light directions and the relative frames.
        """
        
        # Create methods to perform feature matching
        self.sift = cv.SIFT_create(nfeatures=1000)
        self.flann = cv.FlannBasedMatcher_create()
        
        self.params = Parameters()
        
        pass

    def getPointFromImage(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            params[0].append(np.array((x, y), dtype=np.float32))
            print(f"Point: ({x}, {y})")
        pass
    
    def getWorldHomographyFromPts(self, pts, defaultSize):
        """The function computes the projective transformation (homography) between a set of four points choosen from the image plane and a set of destination points denoting the world space.

        Args:
            video (Video): Video class instance, with the video to get the world homography.

        Returns:
            Any: A Matrix represeting the Homograhy if the operation is successfull. An empty list, otherwise.
        """

        # Array used to store points, to then use them to calculate the homography
        points = pts

        # Set the destination points for the real world.
        # In this case we are setting to project the image into a square
        destinationPoints = np.array([
            [0, 0],
            [defaultSize, 0],
            [defaultSize, defaultSize],
            [0, defaultSize]
        ])
        
        # Convert the points into integer value
        points = np.array(points).astype(np.int32)
        
        try:
            # Now we compute the Homography between the World system and the Static Camera
            homograhy, _ = cv.findHomography(points, destinationPoints)
            
            # Store the points for future purpose
            self.points = points

            # Return the homography, even if not defined (None)
            return homograhy
        except:
            return None

    def getWorldHomography(self, video: Video):
        """The function computes the projective transformation (homography) between a set of four points choosen from the image plane and a set of destination points denoting the world space.

        Args:
            video (Video): Video class instance, with the video to get the world homography.

        Returns:
            Any: A Matrix represeting the Homograhy if the operation is successfull. An empty list, otherwise.
        """
        
        # Set video to initial frame, in order to get it
        video.setVideoFrame()

        # ... and get the first frame
        _, frame = video.getCurrentFrame()
        frame = cv.resize(frame, (1080, 1920))

        # Array used to store points, to then use them to calculate the homography
        points = []

        # 3x3 Matrix which will contain the homography between static camera and world
        homograhy = []

        # Now, from the first frame, let's try to compute Homography between World system and the Static Camera
        while True:
            
            # Show the frame and add a mouse callback to get the 4 points
            cv.imshow("Point detection", frame)
            cv.setMouseCallback("Point detection", self.getPointFromImage, param=[points])

            # For simplicity, draw lines between points
            for i in range(len(points)):
                cv.line(frame, tuple(points[i].astype(int)), tuple(points[(i + 1) % len(points)].astype(int)), (0, 0, 255), 3)

            if len(points) == 4:

                # Set the destination points for the real world.
                # In this case we are setting to project the image into a square
                destinationPoints = np.array([
                    [0, 0],
                    [self.params.getOutputImageSize(), 0],
                    [self.params.getOutputImageSize(), self.params.getOutputImageSize()],
                    [0, self.params.getOutputImageSize()]
                ])

                # Convert the points into integer value
                points = np.array(points).astype(np.int32)

                # Now we compute the Homography between the World system and the Static Camera
                homograhy, _ = cv.findHomography(points, destinationPoints)

                # Destroy the window used to retrieve the points
                # cv.destroyWindow("Point detection")
                cv.destroyAllWindows()
                    
                # Press Q on the keyboard to exit.
                if (cv.waitKey(25) & 0xFF == ord('q')):
                    break
                
                # Break inner while since we get them and we computed the Homography
                break    
            
            # Press Q on the keyboard to exit.
            if (cv.waitKey(25) & 0xFF == ord('q')):
                break
        
        self.points = points

        # Return the homography, even if not defined (None)
        return homograhy

    def getPoints(self):
        return self.points

    def extractFeaturesFromFrame(self, frame, idx):
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(grayFrame, None)
        return idx, keypoints, descriptors
    
    def matchFeatures(self, features):
        try:
            features1, features2 = features
            
            idx1, keypoints1, descriptors1 = features1
            _, keypoints2, descriptors2 = features2
                    
            matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        except:
            print("Error in match feature")
            return idx1, None, None, None
            
        src = []
        dst = []
        
        for m1, m2 in matches:
            if m1.distance < 0.7 * m2.distance:
                
                src_pt = keypoints1[m1.queryIdx].pt
                dst_pt = keypoints2[m1.trainIdx].pt
                
                src.append(src_pt)
                dst.append(dst_pt)
                    
        if len(src) >= MIN_MATCH_COUNT:
            
            src = np.float32(src).reshape(-1, 1, 2)
            dst = np.float32(dst).reshape(-1, 1, 2)
            
            homography, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)
            
            return idx1, src, dst, homography
        else:
            return idx1, None, None, None
    
    def getLight(self, staticFrame, movingFrame, homographyStaticToStatic, ptsMovingCam, homographyStaticToMoving, worldHomography, kMoving):  
        
        if homographyStaticToStatic is not None and homographyStaticToMoving is not None:
            
            # Homography mapping points from world reference system to moving camera ref. system
            hWorld2Moving = homographyStaticToMoving @ np.linalg.inv(homographyStaticToStatic) @ np.linalg.inv(worldHomography)
            
            # Homography mapping points from moving camera ref. system to world reference system 
            hMoving2World = worldHomography @ homographyStaticToStatic @  np.linalg.inv(homographyStaticToMoving)
            
            # Option 1: Use a meshgrid to shift points from one ref. system to world ref. system
            # # Create a grid in the moving camera ref. system
            # lx, ly = np.meshgrid(np.linspace(450., 1150., 11), np.linspace(200., 900., 11))   
            # # And plot the points             
            # points2d = np.vstack((lx.flatten(), ly.flatten())).T
            
            # Option 2: Use the features detected in the cam. ref. system and shift points to world ref. system
            points2d = ptsMovingCam
        
            # Add 1 to the source points
            points3d = np.hstack([np.squeeze(points2d), np.ones([points2d.shape[0], 1], dtype=points2d.dtype)])
            
            # Source points inside world reference system
            points3d = hMoving2World @ points3d.T 
            
            points3d /= points3d[2, :]
            
            points3d = points3d.T
            
            # Set last postion to 0
            points3d[:, 2] = 0
            
            # Now get world frame using static camera and homographies to move into the world reference system
            worldFrame = cv.warpPerspective(staticFrame, worldHomography @ homographyStaticToStatic, (self.params.getOutputImageSize(), self.params.getOutputImageSize()))
            # worldFrame = cv.warpPerspective(staticFrame, worldHomography, (self.params.getOutputImageSize(), self.params.getOutputImageSize()))
            
            # ... and do the same for moving camera, in order to get a similarity between frames
            warpedMoving = cv.warpPerspective(movingFrame,  hWorld2Moving, (self.params.getOutputImageSize(), self.params.getOutputImageSize()), flags = cv.WARP_INVERSE_MAP)
        
            
            # Now, let's try to cross-correlate the two warped images.
            # If the correlation is high, then the images are similar, so we can compute the light vector
            # Otherwise, skip the frame
            imgCorr = cv.matchTemplate(worldFrame, warpedMoving, cv.TM_CCORR_NORMED)
                            
            # Set as lower threshold 0.6 to have high confidentiality
            if imgCorr[0][0] >= 0.96:
                # Calculate the light vector using PnP
                
                # Set a treshold (MIN_MATCH_COUNT) which denotes the minimum number of matches to get the Homography
                src = points3d
                dst = np.squeeze(points2d)
                
                if len(src) > MIN_MATCH_COUNT:
                
                    ret, rvec, tvec = cv.solvePnP(src, dst, kMoving, None, flags=cv.SOLVEPNP_IPPE)
                    
                    if not ret:
                        # if solvePnP fails, then return an empty array, corresponding to no light
                        lightVector = worldFrame = None
                    
                    # Get rotation
                    R, _ = cv.Rodrigues(rvec)
                    
                    # then compute light vector
                    lightVector = -R.T @ tvec
                    lightVector = lightVector / np.linalg.norm(lightVector)      
                    
                    # If any of the position is Nan, then skip it
                    if np.isnan(lightVector).any():
                        lightVector = worldFrame = None
                    
                else:
                    lightVector = worldFrame = None
            else:
                lightVector = worldFrame = None
            
        else:
            # Otherwise, if one of the two homographies is not defined, then the light vector is None
            lightVector = worldFrame = None
    
        return worldFrame, lightVector