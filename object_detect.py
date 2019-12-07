
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

import os
import pickle
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
import torch.hub
from gulpio.loader import DataLoader
from gulpio.transforms import ComposeVideo, Scale
from torch.utils.data import DataLoader as DataLoaderPytorch
from tqdm import tqdm

from dataset import VideoDataset, OpticalFlowDataset

parser = ArgumentParser()

parser.add_argument("dataset_path", type=str,
                    help="Path where the dataset is stored.")

parser.add_argument("save_fname", type=str, 
                    help="Filename to give to the final output."
                         "This file will be stored in ./output/")

parser.add_argument("--store_logits", action="store_true", default=False,
                    help="If True, will store the noun and verb logits."
                         "This is useful for ensembling at a later time."
                         "Logits are stored in ./output/basename(dataset_path)/youtube_id.npy")

# This is the maximum number of frames that a single video has in our dataset.
parser.add_argument("--max_frames", type=int, default=1120,
                    help="Maximum number of frames in one video.")

parser.add_argument("--video_batch_size", type=int, default=1,
                    help="Number of videos to read in at one time.")

parser.add_argument("--batch_size", type=int, default=64,
                    help="Number of batches to feed in to the model at one time.")

parser.add_argument("--segment_count", type=int, default=8,
                    help="Number of video segments to focus on at one time.")

parser.add_argument("--snippet_length", type=int, default=1,
                    help="Number of frames per segment.")

parser.add_argument("--optical_flow", default=False, action="store_true",
                    help="If True, will use TSM on Optical Flow data.")

args = parser.parse_args()

# Default height and width for Epic Kitchen models.
HEIGHT, WIDTH = 224, 224
flow_checkpoint_path = "./checkpoints/TSM_arch=resnet50_modality=Flow_segments=8-e09c2d3a.pth.tar"


def batch_it(loader):
    """The dataset loader gives us an output of [num_videos, num_frames, H, W, C]
       The frames are then divided into segments of a specified size and the data 
       ends up looking like this: [num_videos, num_segments, D, H, W]

       This function iterates through every video and yields batch_size segments at 
       one time for it to be processed by the model.
    """
    num_segments = int(args.max_frames / (args.segment_count * args.snippet_length))
    
    pbar = tqdm(total=len(loader))
    for data, num_frames, indices in loader:
        if not args.optical_flow:
            # [num_videos, num_frames, C, H, W]
            data = data.transpose((0, 1, 4, 2, 3))
            data = torch.FloatTensor(data)
        else:
            data = data.float()

        # [num_videos, num_segments, D, H, W]
        data = data.reshape((args.video_batch_size, num_segments, -1, HEIGHT, WIDTH))

        with torch.no_grad():
            num_frames_per_sample = args.segment_count * args.snippet_length

            for b in range(args.video_batch_size):
                index = indices[b]

                if args.optical_flow:
                    youtube_id = os.path.basename(
                                    os.path.dirname(loader.dataset.dataset[index]))
                else:
                    youtube_id = loader.dataset.items[index][0]

                num_chunks_left = int(num_frames[b] / num_frames_per_sample)

                for i in range(0, num_segments, args.batch_size):
                    # [B, D, H, W]
                    # Notice, that the size of the batch may vary from iteration 
                    # to iteration. Thus the memory the GPU is registering will 
                    # fluctuate as we go from small videos to large ones.
                    chunk = data[b, i:i+args.batch_size].squeeze(0).cuda()

                    if chunk.size(0) >= num_chunks_left:
                        yield chunk[:num_chunks_left], youtube_id
                        break
                    else:
                        yield chunk, youtube_id
                        num_chunks_left -= args.batch_size

        pbar.update(1)


def main():

    if args.optical_flow:
        dataset = OpticalFlowDataset(args.dataset_path,
                                     num_frames=args.max_frames,
                                     step_size=1)

        loader = DataLoaderPytorch(dataset,
                                   batch_size=args.video_batch_size,
                                   num_workers=0,
                                   shuffle=False)
    else:
        scale = ComposeVideo([Scale((HEIGHT, WIDTH))])

        dataset = VideoDataset(args.dataset_path, 
                            num_frames=args.max_frames,
                            step_size=1, 
                            is_val=True,
                            transform=scale, 
                            stack=True,
                            random_offset=False)

        loader = DataLoader(dataset, 
                            batch_size=args.video_batch_size, 
                            num_workers=0, 
                            shuffle=False)


    repo = "epic-kitchens/action-models"
    class_counts = (125, 352)
    base_model = "resnet50"

    t = "RGB" if not args.optical_flow else "Flow"
    model = torch.hub.load(repo, "TSM", 
                          class_counts, args.segment_count, t,
                          base_model=base_model, 
                          pretrained="epic-kitchens").cuda()


    try:
        if args.optical_flow:
            checkpoint = torch.load(flow_checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
    except:
        print("Unable to load TSM Optical Flow checkpoint.")
        print("Please download it from: https://data.bris.ac.uk/data/dataset/2tw6gdvmfj3f12papdy24flvmo")
        return 1


    logits_dir = os.path.join("./output/", 
                              os.path.basename(os.path.abspath(args.dataset_path)))
    last_id, last_logits = None, []

    if args.store_logits and not os.path.isdir(logits_dir):
        os.mkdir(logits_dir)

    results = defaultdict(list)

    for chunk, youtube_id in batch_it(loader):

        features = model.features(chunk)
        verb_logits, noun_logits = model.logits(features)

        verb_logits_cpu, noun_logits_cpu = verb_logits.cpu(), noun_logits.cpu()

        verbs = verb_logits_cpu.argmax(dim=1).numpy().tolist()
        nouns = noun_logits_cpu.argmax(dim=1).numpy().tolist()

        results[youtube_id].extend(list(zip(verbs, nouns)))

        if args.store_logits:
            if last_id is None:
                last_id = youtube_id
                last_logits = [verb_logits_cpu.numpy(), noun_logits_cpu.numpy()]
                continue

            if last_id != youtube_id:
                np.save(os.path.join(logits_dir, "{}_verb.npy".format(last_id)), last_logits[0])
                np.save(os.path.join(logits_dir, "{}_noun.npy".format(last_id)), last_logits[1])

                last_id = youtube_id
                last_logits = [verb_logits_cpu, noun_logits_cpu]
            else:
                last_logits[0] = np.concatenate((last_logits[0], verb_logits_cpu))
                last_logits[1] = np.concatenate((last_logits[1], noun_logits_cpu))

    pickle.dump(results, 
                open(os.path.join("./output/", args.save_fname), "wb"))

    if args.store_logits:
        np.save(os.path.join(logits_dir, "{}_verb.npy".format(last_id)), last_logits[0])
        np.save(os.path.join(logits_dir, "{}_noun.npy".format(last_id)), last_logits[1])

    return 0


if __name__ == "__main__":
    sys.exit(main())
