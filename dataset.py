"""Most of this code is taken from the GulpVideoDataset class found in: 
        https://github.com/TwentyBN/GulpIO/blob/master/src/main/python/gulpio/transforms.py
"""

import os

import numpy as np
from gulpio import GulpDirectory

from torch.utils.data import Dataset


class VideoDataset(object):

    def __init__(self, data_path, num_frames, step_size,
                 is_val, transform=None, stack=True, random_offset=True):
        r"""Simple data loader for GulpIO format.
            Args:
                data_path (str): path to GulpIO dataset folder
                num_frames (int): number of frames to be fetched.
                step_size (int): number of frames skippid while picking
            sequence of frames from each video.
                is_val (bool): sets the necessary augmention procedure.
                transform (object): set of augmentation steps defined by
            Compose(). Default is None.
                stack (bool): stack frames into a numpy.array. Default is True.
                random_offset (bool): random offsetting to pick frames, if
            number of frames are more than what is necessary.
        """
        self.gd = GulpDirectory(data_path)
        self.items = list(self.gd.merged_meta_dict.items())
        self.num_chunks = self.gd.num_chunks

        if self.num_chunks == 0:
            raise(GulpIOEmptyFolder("Found 0 data binaries in subfolders " +
                                    "of: ".format(data_path)))

        print(" > Found {} chunks".format(self.num_chunks))
        self.data_path = data_path
        self.transform_video = transform
        self.num_frames = num_frames
        self.step_size = step_size
        self.is_val = is_val
        self.stack = stack
        self.random_offset = random_offset


    def __getitem__(self, index):
        """
        With the given video index, it fetches frames. This functions is called
        by Pytorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        """
        item_id, item_info = self.items[index]

        frames = item_info['frame_info']
        num_frames = len(frames)
        # set number of necessary frames
        if self.num_frames > -1:
            num_frames_necessary = self.num_frames * self.step_size
        else:
            num_frames_necessary = num_frames

        offset = 0
        if num_frames_necessary < num_frames and self.random_offset:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)

        # set target frames to be loaded
        frames_slice = slice(offset, num_frames_necessary + offset,
                             self.step_size)
        frames, meta = self.gd[item_id, frames_slice]

        # padding last frame
        if num_frames_necessary > num_frames:
            # Pad last frame if video is shorter than necessary
            frames_needed = int((num_frames_necessary - num_frames) / self.step_size)
            frames.extend([frames[-1]] * frames_needed)

        # augmentation
        if self.transform_video:
            frames = self.transform_video(frames)

        # format data to torch tensor
        if self.stack:
            frames = np.stack(frames)

        return frames, num_frames, index


    def __len__(self):
        """
        This is called by PyTorch dataloader to decide the size of the dataset.
        """
        return len(self.items)



class OpticalFlowDataset(Dataset):

    def __init__(self, data_path, num_frames, step_size,
                 transform=None):
        """Data Loader for the Optical Flow dataset.
            data_path (str): path to GulpIO dataset folder
            num_frames (int): number of frames to be fetched.
            step_size (int): number of frames skippid while picking
            transform (object): set of augmentation steps defined by
        """
        self.data_path = data_path
        self.num_frames = num_frames
        self.step_size = step_size
        self.transform = transform

        flow_files = [] 
        for root, subdirs, files in os.walk(self.data_path):
            for f in files:
                if f.endswith('.npy'):
                    flow_files.append(os.path.join(root, f))

        self.dataset = flow_files


    def __getitem__(self, index):
        # F x 2 x H x W
        data = np.load(self.dataset[index])

        num_frames, _, H, W = data.shape
        num_frames_necessary = self.num_frames * self.step_size

        if num_frames_necessary > num_frames:
            frames_needed = int((num_frames_necessary - num_frames) / self.step_size)

            new_data = np.zeros((num_frames + frames_needed, 2, H, W))
            new_data[:num_frames] = data
            new_data[num_frames:] = data[-1]

            data = new_data

        if self.transform:
            for frame in range(num_frames):
                data[frame] = self.transform(data[frame])

        return data, num_frames, index

    def __len__(self):
        return len(self.dataset)
