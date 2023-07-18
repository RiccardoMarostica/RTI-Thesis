import numpy as np
import moviepy.editor as mp
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1.2_5/bin/ffmpeg"

from constants import *
from classes.video import Video
from classes.parameters import Parameters

class VideoSynchronisation:
    """This class is used to synchronise the two videos looking at the object of interest (static camera and moving camera). In fact, the two videos are not temporally synchronised, meaning that the first frame of the first video does not correspond to the first frame of the second video.\n
    To perform video synchonisation it's necessary to calculate the time skew between two videos. From the time skew, then it's possible to calculate the frame difference beteween the two videos.\n
    To calculate the time skew, audio sampling is extracted and used to detect the difference between the two tracks.
    """
    
    def __init__(self, path1 : str, path2 : str) -> None:
        """The constructor takes the file path of the two videos, to extract their audio and start with synchronisation

        Args:
            path1 (str): File path of the first video
            path2 (str): File path of the second video
        """
        # Through MoviePy get audio file clip from both video files using the relative paths
        self.audio1 = mp.AudioFileClip(path1)
        self.audio2 = mp.AudioFileClip(path2)
        
        self.params = Parameters()
            
        # Get default audio sampling rate
        self.audio_sampling_rate = self.params.defaultBitRate
        
        # Suppose at the beginning the offset is 0
        self.offset = 0
        
    def synchroniseVideo(self) -> None:
        """The function calculates the offset time between the two videos. The offset between the videos is calculated looking at their audio track and computing the time skew using cross-correlation.\n
        Since the audio is stereo (meaning two audio channels, left and right), cross-correlation is computed on both channels, and the best one is choosen as the time skew.
        """
        
        # Get first 5 seconds to compute the offset between the two audio
        audio1 = self.audio1.subclip(0, 5).to_soundarray(fps = self.audio_sampling_rate)
        audio2 = self.audio2.subclip(0, 5).to_soundarray(fps = self.audio_sampling_rate)
                
        # Compute the cross-correlation of the left and right audio channels
        corr_left = np.correlate(audio1[:, 0], audio2[:, 0], mode='full')
        corr_right = np.correlate(audio1[:, 1], audio2[:, 1], mode='full')
                
        # Then, calculate the average of the audio using both left and right channels
        audio1Avg = (audio1[:, 0] + audio1[:, 1]) / 2
        audio2Avg = (audio2[:, 0] + audio2[:, 1]) / 2
        
        # ... and from them calculate the average cross-correlation
        corr_avg = np.correlate(audio1Avg, audio2Avg, mode="full")
                
        # Now, find the sample offset that maximizes the cross-correlation
        offsets = np.arange(-len(audio2) + 1, len(audio1))
        
        # Get the offset of the left channel and the right channel
        offset_left = offsets[np.argmax(corr_left)]
        offset_right = offsets[np.argmax(corr_right)]
        
        # Then compute the average offset
        offset_avg = offsets[np.argmax(corr_avg)]
        
        # Choose the best alignment based on the highest correlation value, and store it as class field
        if np.max(corr_left) >= np.max(corr_right) and np.max(corr_left) >= np.max(corr_avg):
            self.offset = offset_left
        elif np.max(corr_right) >= np.max(corr_left) and np.max(corr_right) >= np.max(corr_avg):
            self.offset = offset_right
        else:
            self.offset = offset_avg

    
    def getOffset(self):
        return self.offset
    
    def getFrameDifference(self, fps : int) -> int:
        """The functions returns the frame difference between the two videos.

        Args:
            fps (int): Frame per Second used to calculate the frame difference

        Returns:
            int: Frame difference between the two videos
        """
        # Calucate the offset in seconds
        offset_sec = self.offset / self.audio_sampling_rate
        # Then multiply by the number of fps to obtain frame differences.
        # Careful here, the returned value can be positive or negative.
        # If the result is positive, then the first video needs to be shifted
        # Otherwise, shift the second video
        return int(round(offset_sec * fps))