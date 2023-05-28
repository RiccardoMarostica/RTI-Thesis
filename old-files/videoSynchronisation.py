# Import for Moviepy
import numpy as np
import cv2 as cv
import moviepy.editor as mp
import os
import time
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1.2_5/bin/ffmpeg"

from constants import *

# Load the two videos and extract their audio
video1 = cv.VideoCapture(STATIC_VIDEO_FILE_PATH)
audio1 = mp.AudioFileClip(STATIC_VIDEO_FILE_PATH)

video2 = cv.VideoCapture(MOVING_VIDEO_FILE_PATH)
audio2 = mp.AudioFileClip(MOVING_VIDEO_FILE_PATH)

# Now, get the inital 5 seconds in order to calculate the offset.
# From these 5 seconds, retrieve the sound array and convert their values using as sampling rate 44100 Hz.
# That value represnts the best value for audio sampling.
audio1 = audio1.subclip(0, 5).to_soundarray(fps=DEFAULT_SAMPLING_AUDIO_RATE)
audio2 = audio2.subclip(0, 5).to_soundarray(fps=DEFAULT_SAMPLING_AUDIO_RATE)

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
    offset = offset_left
elif np.max(corr_right) >= np.max(corr_left) and np.max(corr_right) >= np.max(corr_avg):
    offset = offset_right
else:
    offset = offset_avg

# Then, get the fps of both video, and check if they are equal
fpsVideo1 = video1.get(cv.CAP_PROP_FPS)
fpsVideo2 = video2.get(cv.CAP_PROP_FPS)
    
# Get the default FPS, using the highest one
defaultFps = max(fpsVideo1, fpsVideo2)

# Get offset in seconds
offset_sec = offset / DEFAULT_SAMPLING_AUDIO_RATE

# Now, we can compute the frame count to shift the two videos
frameCount = abs(int(round(offset_sec * defaultFps)))

print("Frame difference: ", frameCount)

if offset > 0:
    # If the offset is positive, then the first video starts sooner.
    # So move its position in order to start as the second video
    print("First case")

    video1.set(cv.CAP_PROP_POS_FRAMES, frameCount)
    video2.set(cv.CAP_PROP_POS_FRAMES, 0)
elif offset < 0:
    # Otherwise, if the offset is negative, then we have the opposite scenario.
    print("Second case")
    
    video1.set(cv.CAP_PROP_POS_FRAMES, 0)
    video2.set(cv.CAP_PROP_POS_FRAMES, frameCount)

previousTime = 0

# While one of the two videos is open, then read frame by frame
while video1.isOpened() or video2.isOpened():

    time_elapsed = time.time() - previousTime
    
    # Get the frame from each video
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    # If one of the two does not return a frame, then exit the loop
    if not ret1 or not ret2:
        break
    
    if time_elapsed > 1./DEFAULT_FPS_RATE:
        previousTime = time.time()
    
        cv.imshow("Frames Video 1", frame1)
        cv.imshow("Frames Video 2", frame2)

    # Press Q on the keyboard to exit.
    if (cv.waitKey(25) & 0xFF == ord('q')):
        break

# Release videos
video1.release()
video2.release()

# And destroy windows
cv.destroyAllWindows()