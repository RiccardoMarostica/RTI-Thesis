import numpy as np
import moviepy.editor as mp
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1.2_5/bin/ffmpeg"

from constants import *
from classes.video import Video

class VideoSynchronisation:
    
    audio_sampling_rate = DEFAULT_SAMPLING_AUDIO_RATE
    
    def __init__(self, path1, path2) -> None:
        # Through MoviePy get audio file clip from both video files using the relative paths
        self.audio1 = mp.AudioFileClip(path1)
        self.audio2 = mp.AudioFileClip(path2)
        
        # Suppose at the beginning the offset is 0
        self.offset = 0
        
    def synchroniseVideo(self) -> None:
        # Get only the first 5 seconds to compute the shift between the two audio
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
        
        # Choose the best alignment based on the highest correlation value
        if np.max(corr_left) >= np.max(corr_right) and np.max(corr_left) >= np.max(corr_avg):
            self.offset = offset_left
        elif np.max(corr_right) >= np.max(corr_left) and np.max(corr_right) >= np.max(corr_avg):
            self.offset = offset_right
        else:
            self.offset = offset_avg

    
    def getFrameDifference(self, fps : int) -> int:
        # Calucate the offset in seconds
        offset_sec = self.offset / self.audio_sampling_rate
        # Then multiply by the number of fps to obtain frame differences.
        # Careful here, the returned value can be positive or negative.
        # If the result is positive, then the first video needs to be shifted
        # Otherwise, shift the second video
        return int(round(offset_sec * fps))