# Import for Moviepy
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1.2_5/bin/ffmpeg"

import moviepy.editor as mp
import cv2 as cv
import numpy as np

# Get the first video, extract the audio and get the audio array of the first 5 seconds
video1 = mp.VideoFileClip("./video/cam-static/coin1.mov")
audio1 : mp.AudioFileClip = video1.audio.subclip(0, 5.0)
audioArray1 = audio1.to_soundarray()

# Repeat the process for the second video
video2 = mp.VideoFileClip("./video/cam-moving/coin1.mp4")
audio2 = mp.AudioFileClip = video2.audio.subclip(0, 5.0)
audioArray2 = audio2.to_soundarray()


video1.close() 
video2.close()

# Now compute cross-correlation between the two audios
correlation = np.correlate(audioArray1[:, 0], audioArray2[:, 0], mode = "same")

# Find the time offset that maximise the cross-correlation
offset = np.argmax(correlation) - len(audioArray1)

# Apply time offset to second video
video2 : mp.VideoFileClip = video2.subclip(max(0, offset), video2.duration + min(0, offset))

print(video2)

# Return the syncronised video
out = cv.VideoWriter("synchronised.mp4", cv.VideoWriter_fourcc(*'mp4v'), video1.fps, (video1.w, video1.h))

for t in range(int(video1.duration * video1.fps)):
    frame1 = video1.get_frame(t / video1.fps)
    frame2 = video2.get_frame(t / video2.fps)
    out.write(np.concatenate((frame1, frame2), axis=1))
out.release()