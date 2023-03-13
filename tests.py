# Import for Moviepy
import numpy as np
import cv2 as cv
import moviepy.editor as mp
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1.2_5/bin/ffmpeg"


# Load the two videos and extract their audio
video1 = cv.VideoCapture('./video/cam-static/coin1.mp4')
audio1 = mp.AudioFileClip('./video/cam-static/coin1.mp4').subclip(0, 5).to_soundarray()

video2 = cv.VideoCapture('./video/cam-moving/coin1.mp4')
audio2 = mp.AudioFileClip('./video/cam-moving/coin1.mp4').subclip(0, 5).to_soundarray()

# Compute the cross-correlation of the left audio channels
corr = np.correlate(audio1[:, 0], audio2[:, 0], mode='full')
offset = corr.argmax() - (len(corr) // 2)
print(f"Audio offset: {offset}")

# Get offset in seconds (as an integer)
offset_sec = int(round(offset / 44100))
print(f"Audio offset in seconds: {offset_sec}")

# Now, we can compute the frame count to shift the two videos
frameCount = int(offset_sec * int(round(video1.get(cv.CAP_PROP_FPS))))

if offset > 0:
    print("First case")
    video1.set(cv.CAP_PROP_POS_FRAMES, frameCount)
    video2.set(cv.CAP_PROP_POS_FRAMES, 0)
elif offset < 0:
    print("Second case")
    video1.set(cv.CAP_PROP_POS_FRAMES, 0)
    video2.set(cv.CAP_PROP_POS_FRAMES, -frameCount)

while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    
    if not ret1 or not ret2:
        break
    
    cv.imshow("Frame video 1", frame1)
    cv.imshow("Frame video 2", frame2)

    # Press Q on the keyboard to exit.
    if (cv.waitKey(25) & 0xFF == ord('q')):
        break 
        
video1.release()
video2.release()

cv.destroyAllWindows()

##
# Steps da effettuare per eseguire la sincronizzazione dei video.
# 1 - Ottenere i due video, e le loro informazioni.
# 2 - Impostare entrambi i video con la durata del video più corto. Reimpostare poi il puntatore all'inizio del video
# 3 - Una volta che i due video sono della stessa lunghezza, calcolare l'offset che c'è tra i due in base all'audio
# 4 - Capire a quale dei due video è necessario effettuare l'offset
# 5 - applicare l'offset al video corretto
# #
# video1 = cv.VideoCapture("./video/cam-static/coin1.mp4")
# video2 = cv.VideoCapture("./video/cam-moving/coin1.mp4")

# # Get the frame per second of each video
# # We assume they both have the same frame per second
# fps1 = int(round(video1.get(cv.CAP_PROP_FPS)))
# fps2 = int(round(video2.get(cv.CAP_PROP_FPS)))

# # Get the number of frames in both video
# frameCount1 = int(video1.get(cv.CAP_PROP_FRAME_COUNT))
# frameCount2 = int(video2.get(cv.CAP_PROP_FRAME_COUNT))

# # From the previous two information, compute the duration in seconds
# videoLength1 = int(round(frameCount1 / fps1))
# videoLength2 = int(round(frameCount2 / fps2))

# print("Frame count video 1: ", frameCount1)
# print("Frame count video 2: ", frameCount2)

# # Chekc which video is the smallest, and store its frame count
# if (videoLength1 <= videoLength2):
#     frameCount = frameCount1
# else:
#     frameCount = frameCount2

# print("Frame count: ", frameCount)

# # Set the pointer for both video at the beginning (setting position frame to zero)
# video1.set(cv.CAP_PROP_POS_FRAMES, 0)
# video2.set(cv.CAP_PROP_POS_FRAMES, 0)

# # ... and set the total number of frames for both video to the frame counts
# video1.set(cv.CAP_PROP_FRAME_COUNT, frameCount)
# video2.set(cv.CAP_PROP_FRAME_COUNT, frameCount)

# while True:
#     ret1, frame1 = video1.read()
#     ret2, frame2 = video2.read()

#     if not ret1 or not ret2:
#         break
    
#     cv.imshow("Frame video 1", frame1)
#     cv.imshow("Frame video 2", frame2)

#     # Press Q on the keyboard to exit.
#     if (cv.waitKey(25) & 0xFF == ord('q')):
#         break
    
# video1.release()
# video2.release()

# # ... and closes all windows
# cv.destroyAllWindows()