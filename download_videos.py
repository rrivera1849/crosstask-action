import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

import csv
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool

import pytube
from pytube import YouTube

parser = ArgumentParser()

parser.add_argument("videos_csv", type=str,
                    help="CSV file that contains all the video URL's.")

parser.add_argument("save_to", type=str,
                    help="Folder to save video files in.")

parser.add_argument("--resolution", default=None, type=str,
                    help="Resolution at which to download the videos."
                         "If None, downloads the video at the highest resolution.")

parser.add_argument("--type", default="mp4", type=str,
                    help="Mime type to download, defaults to MP4.")

parser.add_argument("--num_workers", type=int, default=40,
                    help="Number of workers to use while downloading the videos.")

args = parser.parse_args()


def read_csv(video_file):
    """Reads the CSV file specified. 
       We assume that each row has three elements:
            1. Task ID
            2. YouTube ID
            3. URL
    """
    results = []

    with open(video_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            results.append((row[0], row[1], row[2]))

    return results


def download_video(csv_line):
    task_id, youtube_id, url = csv_line

    fname = "{}_{}".format(task_id, youtube_id)

    try:
        yt = YouTube(url)
    except Exception as e:
        print(e, url)
        fname = "{}_{}.empty".format(task_id, youtube_id)
        open(os.path.join(args.save_to, fname), 'a').close()
        return 

    if args.resolution:
        query = yt.streams.filter(progressive=True, file_extension="mp4", resolution=args.resolution).first()
    else:
        query = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()

    if os.path.exists(os.path.join(args.save_to, fname)):
        return

    query.download(output_path=args.save_to, filename=fname)


def main():
    csv_lines = read_csv(args.videos_csv)

    pool = Pool(args.num_workers)
    pool.map(download_video, csv_lines)
    pool.close()

    return 0

if __name__ == "__main__":
    if not os.path.isdir(args.save_to):
        parser.error("save_to must point to a directory.")

    sys.exit(main())
