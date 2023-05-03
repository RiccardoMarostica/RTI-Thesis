# Import for Moviepy
import numpy as np
import cv2 as cv
import moviepy.editor as mp
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1.2_5/bin/ffmpeg"

from constants import *

# Load the two videos and extract their audio
video1 = cv.VideoCapture('./video/cam-static/paper.mp4')
audio1 = mp.AudioFileClip('./video/cam-static/paper.mp4')

video2 = cv.VideoCapture('./video/cam-moving/paper.mp4')
audio2 = mp.AudioFileClip('./video/cam-moving/paper.mp4')

# Now, get the inital 5 seconds in order to calculate the offset.
# From these 5 seconds, retrieve the sound array and convert their values using as sampling rate 44100 Hz.
# That value represnts the best value for audio sampling.
audio1 = audio1.subclip(0, 5).to_soundarray(fps=DEFAULT_SAMPLING_AUDIO_RATE)
audio2 = audio2.subclip(0, 5).to_soundarray(fps=DEFAULT_SAMPLING_AUDIO_RATE)

# Compute the cross-correlation of the left audio channels
corr = np.correlate(audio1[:, 0], audio2[:, 0], mode='full')
offset = corr.argmax() - (len(corr) // 2)

# Get offset in seconds
offset_sec = offset / DEFAULT_SAMPLING_AUDIO_RATE

print("Offset in seconds: ", offset_sec)

# Then, get the fps of both video, and check if the both are equal to the default fps rate
fpsVideo1 = int(round(video1.get(cv.CAP_PROP_FPS)))
fpsVideo2 = int(round(video2.get(cv.CAP_PROP_FPS)))

# If one of them is not as the default fps rate, then modify the video
if (fpsVideo1 != DEFAULT_FPS_RATE or fpsVideo2 != DEFAULT_FPS_RATE):
    # TODO: Aggiungere parte per modifica FPS del video
    print("Different Video FPS")

# Now, we can compute the frame count to shift the two videos
frameCount = abs(int(round(offset_sec * DEFAULT_FPS_RATE)))

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

# While one of the two videos is open, then read frame by frame
while video1.isOpened() or video2.isOpened():
    
    # Get the frame from each video
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    # If one of the two does not return a frame, then exit the loop
    if not ret1 or not ret2:
        break
    
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