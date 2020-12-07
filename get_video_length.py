from cv2 import cv2
import sys
import time
import os


def get_video_capture(vid_filepath):

    if not os.path.exists(vid_filepath):
        print("Video filename not found")

    vid = cv2.VideoCapture(vid_filepath)
    while not vid.isOpened():
        time.sleep(1)

    frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    return frames


def main():
    filename = sys.argv[1]
    frames = get_video_capture(filename)
    print(frames)


if __name__ == "__main__":
    main()
