class Parameters:
    
    # Fields - Parameters 
    # Calibration video paths
    STATIC_VIDEO_CALIBRATION_FILE_PATH = ""
    MOVING_VIDEO_CALIBRATION_FILE_PATH = ""
    # Video Analysis paths
    STATIC_VIDEO_FILE_PATH = ""
    MOVING_VIDEO_FILE_PATH = ""
    
    def __init__(self) -> None:
        pass
    
    def getStaticCameraAnalysisPath(self) -> str:
        return self.STATIC_VIDEO_FILE_PATH
    
    def getMovingCameraAnalysisPath(self) -> str:
        return self.MOVING_VIDEO_FILE_PATH
    
    def getStaticCameraCalibrationPath(self) -> str:
        return self.STATIC_VIDEO_CALIBRATION_FILE_PATH
    
    def getMovingCameraCalibrationPath(self) -> str:
        return self.MOVING_VIDEO_CALIBRATION_FILE_PATH
    
    def setStaticCameraAnalysisPath(self, path: str) -> None:
        self.STATIC_VIDEO_FILE_PATH = path
    
    def setMovingCameraAnalysisPath(self, path: str) -> None:
        self.MOVING_VIDEO_FILE_PATH = path
    
    def setStaticCameraCalibrationPath(self, path: str) -> None:
        self.STATIC_VIDEO_CALIBRATION_FILE_PATH = path
    
    def setMovingCameraCalibrationPath(self, path: str) -> None:
        self.MOVING_VIDEO_CALIBRATION_FILE_PATH = path