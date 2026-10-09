import sys
import os
import cv2

# Intercept VideoCapture to route V4L2 calls to Windows DirectShow
_native_videocapture = cv2.VideoCapture

def windows_videocapture(*args, **kwargs):
    args_list = list(args)
    if len(args_list) > 1 and args_list[1] == cv2.CAP_V4L2:
        args_list[1] = cv2.CAP_DSHOW
    elif len(args_list) == 1:
        args_list.append(cv2.CAP_DSHOW)
    return _native_videocapture(*args_list, **kwargs)

cv2.VideoCapture = windows_videocapture

# Force 720p resolution in memory
import config
config.CAPTURE_WIDTH = 1280
config.CAPTURE_HEIGHT = 720

from app import HearMeApp

if __name__ == "__main__":
    print("[✓] Running natively on Windows (720p DirectShow mode)")
    app = HearMeApp()
    app.mainloop()
