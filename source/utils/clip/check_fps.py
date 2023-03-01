import cv2
import os

#path="/dataset/AIC2022/1/VideoC01_V01"
path = "/dataset/AIC2022/0/VideoC00_V00/C00_V0000.mp4"

def check_fps(path):
    """for sub_f in os.listdir(path):
        if 'mp4' not in sub_f:
            continue
        sub_path = os.path.join(path, sub_f)"""
    video = cv2.VideoCapture(path)  #sub_path
    fps = round(video.get(cv2.CAP_PROP_FPS))
    print(fps)

check_fps(path)