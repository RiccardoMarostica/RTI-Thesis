import cv2 as cv
import numpy as np
from scipy.interpolate import Rbf
from constants import *

from classes.video import Video

class LightDirection:
    def __init__(self, frame, staticFrame, lightVector) -> None:
        self.frame = frame
        self.staticFrame = staticFrame
        self.ligthVector = lightVector
        pass

class RTI:
    def __init__(self) -> None:
        """The constructor inistalise SIFT to get features from a frame, and FLANN for feature matching.
        Also, it initialise a list used to store light directions and the relative frames.
        """
        
        # Create methods to perform feature matching
        self.sift = cv.SIFT_create()
        self.flann = cv.FlannBasedMatcher_create()
        self.bruteforce = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        
        self.lightDirections : list[LightDirection] = []
        pass

    def getDefaultK(self, video : Video):
        """This function returns a default Intrinsic Parameters Matrix K.
        It sets speficic values for focal lenght (fx, fy) and optical centers (cx,cy).
        In this case, fx and fy are the height of the Video instance, while cx and cy are half of the width and height of the Video.

        Args:
            video (Video): Video class instance denoting the Video

        Returns:
            NDArray[float32]: A 3 x 3 matrix representing a default Intrinsic Parameters Matrix
        """
        
        # Get the video width and height, so we can get cx, cy, fx and fy
        width = int(video.getWidth())
        height = int(video.getWidth())

        cx = width // 2   # Get the integer value of cx
        cy = height // 2  #  Get the integer value of cy

        fx = fy = height  # Get the integer value of fx, fy

        # Build K and return it
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32) 

    def getPointFromImage(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            params[0].append(np.array((x, y), dtype=np.float32))
            print("Point added")
        pass

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
                    [DEFAULT_SQUARE_SIZE, 0],
                    [DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE],
                    [0, DEFAULT_SQUARE_SIZE]
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


        # Return the homography, even if not defined (None)
        return homograhy
        # return [] if homograhy is None else homograhy

    '''
    H -> Homography
    K -> Intrinsic Parameters of the camera
    The function computes the extrinsic parameters using the homography and intrinsic parameters array.
    At the end, it returns the derived rotation and translation vectors
    '''
    def getExtrinsicsParameters(self, H, K) -> tuple:
        """The function estimates the Camera Pose of the Camera. From an Homography H and the Instrinsic Parameters K, computes the Rotation Matrix R and the Translation Vector T, the two elements used to get the Camera Pose.

        Args:
            H (Any): Homography between two views
            K (Any): Instrinsic Parameters Matrix of a Camera

        Returns:
            tuple: Returns R (Rotation Matrix) and T (Translation Vector)
        """
        
        H = H.T                                             # Transpose of H
        
        K_inverse = np.linalg.inv(K)                        # Inverse of K
        
        h1 = H[0]                                           # First column
        h2 = H[1]                                           # Second column
        h3 = H[2]                                           # Third column
        
        alpha = 1 / np.linalg.norm(np.dot(K_inverse, h1))   # Scale factor
        
        r1 = alpha * np.dot(K_inverse, h1)                  #  Rotation matrix first column
        r2 = alpha * np.dot(K_inverse, h2)                  #  Rotation matrix second column
        r3 = np.cross(r1, r2)                               #  Rotation matrix third column
        
        
        T = alpha * (K_inverse @ h3.reshape(3, 1))          # Get the translation vector
        R = np.array([[r1], [r2], [r3]])                    #  Get the rotation matrix
        R = np.reshape(R, (3, 3))

        return R, T

    def getHomographyWithFeatureMatching(self, frame1, frame2):
        """The function retrieves an homography between two views, trough feature matching.\n
        For both views (two distinct frames), features are detected using SIFT. The detected features (with keypoints and descriptors) are matched in the two views using FLANN matcher, in which for each descriptor K best matches are found.\n
        From the matches, then, the keypoints of both views (source view as frame1 and destination view as frame2) are extracted and from them the homography between the two views is calculated.

        Args:
            frame1 (Any): Frame representing the view of an object from a camera
            frame2 (Any): Frame representing the view of an object from a camera

        Returns:
            Any: Return a projective transformation (Homography - 3x3 Matrix) if the feature matching is done correctly, otherwise returns an empty list.
        """
        
        try:
            # Compute features using SIFT in both frames. Return keypoints and related descriptors
            keypoints1, descriptors1 = self.sift.detectAndCompute(frame1, None)
            keypoints2, descriptors2 = self.sift.detectAndCompute(frame2, None)

            # Feature matching using KNN (K-Nearest-Neighborhood) technique of FLANN
            matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
            # matches = self.bruteforce.match(descriptors1, descriptors2)
        except:
            # If an error occurs in the calculation of the matches, just return an empty array corresponding to empty homography
            return []
        
        src = []
        dst = []
        
        goodMatches = []
        for m1, m2 in matches:
            if m1.distance < 0.7 * m2.distance:
                src.append(keypoints1[m1.queryIdx].pt)
                dst.append(keypoints2[m1.trainIdx].pt)
                goodMatches.append(m1)

        # Set a treshold (MIN_MATCH_COUNT) which denotes the minimum number of matches to get the Homography
        if len(src) > MIN_MATCH_COUNT:
            # Get source and destination points found inside the good matches to build the homography between the two frames
            src = np.float32(src).reshape(-1, 1, 2)
            dst = np.float32(dst).reshape(-1, 1, 2)
            
            # Get the Homography. In this case the method used to findthe transformation is through RANSAC, a consensus-based approach. Since RANSAC is used, it's necessary to set a treshold in which a point pair is considered as an inlier.
            homography, _ = cv.findHomography(src, dst, cv.RANSAC, 5.0)     
                        
            # Draw matches
            img_matches = np.empty((max(frame1.shape[0], frame2.shape[0]), frame1.shape[1]+frame2.shape[1], 3), dtype=np.uint8)
            cv.drawMatches(frame1, keypoints1, frame2, keypoints2, goodMatches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv.imshow('Good Matches', img_matches)
            
            # Press Q on the keyboard to exit.
            if (cv.waitKey(25) & 0xFF == ord('q')):
                return homography
            
                   
            return homography
        else:
            return []
        
    def getLightVector(self, R, T):
        """The function returns an estimated light vector, using the Camera Pose.
        In fact, assuming that the flash light of a Camera corresponds to the optical center, it's possible to obtain the optical center as O = -R.T * T.
        Getting this, it's possible to divide it by it's norm and obtain the light vector.

        Args:
            R (Any): Rotation Matrix
            T (Any): Translation Vector

        Returns:
            Any: An list of three elements (X, Y, Z) denoting the light vector for a specific frame, in normalised coordinates.
        """
        R = R.T
        R = -1 * R
        l = np.dot(R, T)
        norm_l = np.linalg.norm(l)
        lightVector = l / norm_l
        return lightVector
        
    def storeLightVector(self, frame, staticFrame, lightVector):
        """The function stores the information about a light vector and the respective frame, which will be used later to perform religthing.
        
        Args:
            frame (Any): Frame related to a specific light vector
            lightVector (Any): Light vector computed from Camera Pose
        """
        self.lightDirections.append(LightDirection(frame, staticFrame, lightVector))
        
    def getLightDirections(self) -> list[LightDirection]:
        """The function returns the list of all the pairs (light vector, related frame) stored during the analysis process.
        
        Returns:
            list[LightDirection]: List of LightDirection class
        """
        return self.lightDirections
            
    def showCircleLightDirection(self, light_direction):
        """The function shows light direction inside a plot.

        Args:
            light_direction (Any): Light direction in normalised coordinates.

        Returns:
            Any: An image representing a plot of the light direction.
        """
        
        # Create a blank image
        image = np.zeros((DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE, 3), dtype=np.uint8)

        center_x = center_y = DEFAULT_SQUARE_SIZE // 2
        radius = DEFAULT_SQUARE_SIZE // 2    
            
        # Draw the circle border
        cv.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)
        
        if len(light_direction) != 0:
            x = int(((light_direction[0][0] + 1) * DEFAULT_SQUARE_SIZE) / 2)
            y = int(((light_direction[1][0] + 1) * DEFAULT_SQUARE_SIZE) / 2)        
            cv.circle(image, (int(x), int(y)), 10, (0, 255, 0), 2)
            cv.line(image, (center_x, center_y), (int(x), int(y)), (0, 255, 0), 2)
        
        return image

    def applRBFInterpolation(self, xf: int, yf: int, nu: int, nv: int):
        """The function (Tensor function) computes RBF Interpolation, using all the images stored in the Analysis step, and the dimension of the space. \n
        The RBF Interpolation for each pixel is stored in a contiguous array, which can be used for relighting.

        Args:
            xf (int): x dimension of the mesh grid which will used when applying interpolation
            yf (int): y dimension of the mesh grid which will used when applying interpolation
            nu (int): u dimension to loop over to get the intensity of a pixel in the stored frames
            nv (int): y dimension to loop over to get the intensity of a pixel in the stored frames
        """
        
        # First use a matrix with size nu x nv, to store the result of interpolation at each position
        self.rbfInterpolation = []
        
        # Then calculate lxf and lyf which are necessary to calculate RBF
        lxf, lyf = np.meshgrid(np.linspace(-1.0, 1.0, xf), np.linspace(-1.0, 1.0, yf))

        # Retrieve the array of stored light directions computed in the previous step
        lightDirections = self.getLightDirections()
             
        # Get each value x and y in the light vector array   
        lx = np.float32([tmp.ligthVector[0] for tmp in lightDirections])
        ly = np.float32([tmp.ligthVector[1] for tmp in lightDirections])
                
        # Now double loop to iterate along all the pixels, get the intensity at the pixel (u, v) from all the stored frames, and calculate RBF Interpolation for each pixel.
        for u in range(nu):
            for v in range (nv):
                i = np.float32([tmp.frame[u, v] for tmp in lightDirections])
                r = Rbf(lx, ly, i, function='linear')
                i_interpolate = r(lxf, lyf)
                self.rbfInterpolation.append(i_interpolate)
                # print("Pixel (", u, ", ", v, ") done")
        pass

    def getRBFInterpolation(self):
        """The function returns the list of Interpolation computed using RBF Interpolation.

        Returns:
            list: List of interpolations
        """
        return self.rbfInterpolation
    
    def findNearestPoint(self, interpolation, value):
        """The function return the nearest point inside the interpolation matrix (xf x yf Matrix).

        Args:
            interpolation (any): Interpolation matrix (with dimension xf x yf)
            value (_type_): coordinate (u or v axis)

        Returns:
            float: Nearest point to the input one.
        """
        # Return the nearest point from value inside the interpolation
        differences = np.abs(interpolation - value)
        min_index = np.unravel_index(differences.argmin(), differences.shape)
        # return nearest point
        return interpolation[min_index]
    
    def findNearestFrame(self, array, input_values):
        """The function returns the index of the best frame for relighting. The index is choosen searching the nearest light vector to the one in input (input_values)

        Args:
            array (Any): Array of light vector
            input_values (Any): Light vector in normalised coordinate

        Returns:
            int: Index of the nearest light vector
        """
        input_values = np.array(input_values)
        distances = np.linalg.norm(array[:, :2, :] - input_values.reshape(1, 2, 1), axis=1)
        nearest_index = np.argmin(distances)
        return nearest_index
    
    def normaliseCoordinate(self, value : float, dim: int) -> float:
        """Convert coordinates to normalised coordinates between -1 and 1

        Args:
            value (float): Coordinate value in the space
            dim (int): Dimension of the axis

        Returns:
            float: Normalised coordiante
        """
        # Convert the coordinates to normalized values between -1 and 1
        return (value / dim) * 2 - 1
    
    def applyRelighting(self):
        """The function show the relighting of the image
        """
        
        center_x = center_y = DEFAULT_SQUARE_SIZE // 2
        radius = DEFAULT_SQUARE_SIZE // 2    

        
        while(True):
            # Draw plot image
            self.relightPlot = np.zeros((DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE, 3), dtype=np.uint8)
            # Draw the circle border
            cv.circle(self.relightPlot, (center_x, center_y), radius, (255, 255, 255), 1)
            cv.line(self.relightPlot, (0, center_y), (DEFAULT_SQUARE_SIZE, center_y), (255, 255, 255), 1)
            cv.line(self.relightPlot, (center_x, 0), (center_x, DEFAULT_SQUARE_SIZE), (255, 255, 255), 1)
            
            
            cv.imshow("Relight plot", self.relightPlot)
            cv.setMouseCallback("Relight plot", self.calculateRelightingFrame, param=[center_x, center_y])
            # Press Q on the keyboard to exit.
            if (cv.waitKey(25) & 0xFF == ord('q')):
                break
            
        return
        
    def calculateRelightingFrame(self, event, x, y, flags, params):
        """The function takes in input an event from the mouse (click on the image) and compute the relighting

        Args:
            event (Any): Event type
            x (int): x coordinate
            y (int): y coordinate
        """
        if event == cv.EVENT_MOUSEMOVE:
            
            # Get information from parameters
            center_x = params[0]
            center_y = params[1]
            
            # Draw the point
            cv.circle(self.relightPlot, (int(x), int(y)), 10, (0, 255, 0), 2)
            cv.line(self.relightPlot, (center_x, center_y), (int(x), int(y)), (0, 255, 0), 2)

            # Get the array containing the information about relighting    
            # rbfInterpolation = self.getRBFInterpolation()
            # ... and light direction array
            lightDirections = self.getLightDirections()
            
            # # Get the interpolation 11 x 11 vector, to search the nearest value
            # interpolationXY = rbfInterpolation[x * DEFAULT_SQUARE_SIZE + y]
            
            # # Compute nearest value for the coordinates X and Y
            # nearest_X = self.findNearestPoint(interpolationXY, x)
            # nearest_Y = self.findNearestPoint(interpolationXY, y)
            nearest_X = x
            nearest_Y = y
            
            # Then compute the normalised coordinates
            norm_X = self.normaliseCoordinate(nearest_X, DEFAULT_SQUARE_SIZE)
            norm_Y = self.normaliseCoordinate(nearest_Y, DEFAULT_SQUARE_SIZE)
            
            print("Normalised coordinates: ", str([norm_X, norm_Y]))
            
            # Recover the lights from the array of light directions
            lights = np.array([tmp.ligthVector for tmp in lightDirections])
                        
            # Get the nearest frame
            index = self.findNearestFrame(lights, [norm_X, norm_Y])
            
            print("Nearest light index: ", str(index))
            print("Nearest light value: ", str(lightDirections[index].ligthVector))
            
            frame = lightDirections[index].frame
            staticFrame = cv.resize(lightDirections[index].staticFrame, (480, 960))
            
            # ... and show it
            cv.imshow("Relighted image", frame)
            cv.imshow("Static frame", staticFrame)
            
            
# TODO:
# 1. Visualizzare al movimento del mouse il cambiamento del frame + direzione del light vector -> FARE DEBUGGING
