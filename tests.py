# Import for Moviepy
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1.2_5/bin/ffmpeg"

import moviepy.editor as mp
import numpy as np

staticCameraClipVideo = mp.VideoFileClip("./video/cam-static/coin1.mov")
staticCameraAudioArray = staticCameraClipVideo.audio.subclip(0, 10).to_soundarray()

staticCameraClipVideo.close()

movingCameraClipVideo = mp.VideoFileClip("./video/cam-moving/coin1.mp4")
movingCameraAudioArray = movingCameraClipVideo.audio.subclip(0, 10).to_soundarray()

movingCameraClipVideo.close()

print(staticCameraAudioArray)