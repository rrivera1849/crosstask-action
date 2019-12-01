
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

import math
import sys
from argparse import ArgumentParser

import torch
import torch.hub
from gulpio.loader import DataLoader
from gulpio.transforms import ComposeVideo, Scale

from dataset import VideoDataset

parser = ArgumentParser()

parser.add_argument("dataset_path", type=str,
                    help="Path where the dataset is stored.")

# This is the maximum number of frames that a single video has in our dataset.
parser.add_argument("--max_frames", type=int, default=1120,
                    help="Maximum number of frames in one video.")

parser.add_argument("--video_batch_size", type=int, default=1,
                    help="Number of videos to read in at one time.")

parser.add_argument("--batch_size", type=int, default=64,
                    help="Number of batches to feed in to the MTRN model at one time.")

parser.add_argument("--segment_count", type=int, default=8,
                    help="Number of video segments to focus on at one time.")

parser.add_argument("--snippet_length", type=int, default=1,
                    help="Number of frames per segment.")

args = parser.parse_args()

# Default height and width for Epic Kitchen models.
HEIGHT, WIDTH = 224, 224


def batch_it(loader):
    """The dataset loader gives us an output of [num_videos, num_frames, H, W, C]
       The frames are then divided into segments of a specified size and the data 
       ends up looking like this: [num_videos, num_segments, D, H, W]

       This function iterates through every video and yields batch_size segments at 
       one time for it to be processed by the model.
    """
    num_segments = int(args.max_frames / (args.segment_count * args.snippet_length))

    for data in loader:
        # [num_videos, num_frames, C, H, W]
        data = data.transpose((0, 1, 4, 2, 3))
        data = torch.FloatTensor(data)

        # [num_videos, num_segments, D, H, W]
        data = data.reshape((args.video_batch_size, num_segments, -1, HEIGHT, WIDTH))

        with torch.no_grad():
            for b in range(args.video_batch_size):
                for i in range(0, num_segments, args.batch_size):
                    # [B, D, H, W]
                    chunk = data[b, i:i+args.batch_size].squeeze(0).cuda()

                    yield chunk


def main():
    repo = "epic-kitchens/action-models"

    scale = ComposeVideo([Scale((HEIGHT, WIDTH))])

    dataset = video_dataset = VideoDataset(args.dataset_path, 
                                           num_frames=args.max_frames,
                                           step_size=1, 
                                           transform=scale, 
                                           is_val=True)

    loader = DataLoader(dataset, batch_size=args.video_batch_size, num_workers=0, shuffle=False)

    class_counts = (125, 352)
    base_model = "resnet50"

    mtrn = torch.hub.load(repo, "MTRN", 
                          class_counts, args.segment_count, "RGB",
                          base_model=base_model, 
                          pretrained="epic-kitchens").cuda()


    num_frames_per_sample = args.segment_count * args.snippet_length
    total = 0

    for chunk in batch_it(loader):
        total += num_frames_per_sample * chunk.size(0)

        features = mtrn.features(chunk)
        verb_logits, noun_logits = mtrn.logits(features)
        print(chunk.size())
        print(verb_logits.shape, noun_logits.shape)

    # assert total == args.max_frames * len(dataset)

    return 0


if __name__ == "__main__":
    sys.exit(main())
