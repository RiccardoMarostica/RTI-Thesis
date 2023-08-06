from classes.cameraCalibration import CameraCalibration

class LightDirection:
    def __init__(self, frame, lightVector) -> None:
        self.frame = frame
        self.ligthVector = lightVector
        pass

class Singleton:
    """Alex Martelli implementation of Singleton (Borg)
    http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Parameters (Singleton):

    defaultBitRate = 44100
    defaultMsecVideoGap = 1500

    # CALIBRATION VARS
    mvCamCalibPath = None
    stCamCalibPath = None
    stCamCalibration = None
    mvCamCalibration = None
    
    # VIDEO ANALYSIS VARS
    mvCamVideoPath = None
    stCamVideoPath = None
    
    # VIDEO VARS
    outputImageSize = 0
    defaultSizeOne = 1920
    defaultSizeTwo = 1080
    
    defaultResizeOne = 960
    defaultResizeTwo = 540
    
    # LIGHT VECTOR
    lightVectors = []

    def __init__(self) -> None:
        Singleton.__init__(self)

        pass

    def getStCamCalibPath(self):
        return self.stCamCalibPath

    def getMvCamCalibPath(self):
        return self.mvCamCalibPath

    def setCamCalibPath(self, camId, path):
        if camId == "stCamBtn":
            # Store the path inside a class variable, which will be used to the calibration method
            self.stCamCalibPath = path

        if camId == 'mvCamBtn':
            # Store the path inside a class variable, which will be used to the calibration method
            self.mvCamCalibPath = path

    def getStCamCalibData(self):
        return self.stCamCalibration

    def getmvCamCalibData(self):
        return self.mvCamCalibration

    def setCamsCalibData(self, camSt: CameraCalibration, camMv: CameraCalibration):
        # Store both calibration class inside the class
        self.stCamCalibration = camSt
        self.mvCamCalibration = camMv

    def setCamVideoPath(self, camId, path):
        if camId == "stCamBtn":
            # Store the path inside a class variable, which will be used to the calibration method
            self.stCamVideoPath = path

        if camId == 'mvCamBtn':
            # Store the path inside a class variable, which will be used to the calibration method
            self.mvCamVideoPath = path
            
    def getStCamVideoPath(self):
        return self.stCamVideoPath

    def getMvCamVideoPath(self):
        return self.mvCamVideoPath
    
    def setOutputImageSize(self, defaultSize):
        self.outputImageSize = defaultSize
        
    def getOutputImageSize(self):
        return self.outputImageSize    
    
    def getFrameDefaultSize(self, type):
        if type == "landscape":
            return (self.defaultSizeOne, self.defaultSizeTwo)
        if type == "portrait":
            return (self.defaultSizeTwo, self.defaultSizeOne)
    
    def getResizedSizePointCapture(self, type):
        if type == "landscape":
            return (self.defaultResizeOne, self.defaultResizeTwo)
        if type == "portrait":
            return (self.defaultResizeTwo, self.defaultResizeOne)
        
    
    def addLightVector(self, light, frame):
        self.lightVectors.append(LightDirection(frame, light))
        
    def getLightVectors(self):
        return self.lightVectors
    