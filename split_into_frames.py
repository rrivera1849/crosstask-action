
import math
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool

import cv2

parser = ArgumentParser()

parser.add_argument("dataset", type=str,
                    help="Path where unprocessed videos are stored.")

parser.add_argument("save_to", type=str,
                    help="Path where processed videos (frames) should be saved.")

args = parser.parse_args()


def split_into_frames(video_path, save_to):
    """Splits the given video into frames (1 FPS) and saves them to the path 
       specified.
    """
    cap = cv2.VideoCapture(video_path)   

    # 1 FPS
    frame_rate = cap.get(5) 

    count = 0
    while (cap.isOpened()):
        frame_id = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break

        if frame_id % math.floor(frame_rate) == 0:
            filename = "frame_{:05d}.jpg".format(count)
            count += 1
            cv2.imwrite(os.path.join(save_to, filename), frame)

    cap.release()


def do_work(video_path):
    base = os.path.basename(video_path)

    all_ = base.split('.')[0].split('_')
    task, youtube_id = all_[0], '_'.join(all_[1:])
    save_to = os.path.join(args.save_to, task, youtube_id)

    if not os.path.isdir(save_to):
        os.makedirs(save_to)

    split_into_frames(video_path, save_to)


def main():
    mp4_files = []
    for root, subdirs, files in os.walk(args.dataset):
        for f in files:
            if f.endswith(".mp4"):
                mp4_files.append(os.path.join(root, f))

    pool = Pool(40)
    pool.map(do_work, mp4_files)
    pool.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
