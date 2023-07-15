from classes.cameraCalibration import CameraCalibration


class Singleton:
    """Alex Martelli implementation of Singleton (Borg)
    http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html"""
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Parameters (Singleton):

    # CALIBRATION VARS
    mvCamCalibPath = None
    stCamCalibPath = None
    stCamCalibration = None
    mvCamCalibration = None

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
