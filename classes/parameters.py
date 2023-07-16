from classes.cameraCalibration import CameraCalibration


class Singleton:
    """Alex Martelli implementation of Singleton (Borg)
    http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Parameters (Singleton):

    defaultBitRate = 44100

    defaultFrameSize = 0

    # CALIBRATION VARS
    mvCamCalibPath = None
    stCamCalibPath = None
    stCamCalibration = None
    mvCamCalibration = None
    
    # VIDEO ANALYSIS VARS
    mvCamVideoPath = None
    stCamVideoPath = None

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

    def getStCamCalibData(self):
        return self.stCamCalibration

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
    
    def setWorldDefaultSize(self, defaultSize):
        self.defaultFrameSize = defaultSize
        
    def getWorldDefaultSize(self):
        return self.defaultFrameSize    
    