
import os
import sys
from collections import defaultdict
from argparse import ArgumentParser

import skimage.io as io
from gulpio.adapters import AbstractDatasetAdapter, Custom20BNAdapterMixin
from gulpio.fileio import GulpIngestor

parser = ArgumentParser()

parser.add_argument("output_folder", type=str,
                    help="Path where Gulp should be stored.")

parser.add_argument("dataset_path", type=str,
                    help="Path where the dataset is stored."
                         "Note: must already be stored as frames.")

parser.add_argument("--annotations_path", type=str, default=None,
                    help="Path where annotations are stored."
                         "Used for storing the metadata.")

parser.add_argument("--videos_per_gulp", type=int, default=100,
                    help="Number of videos to store per Gulp.")

parser.add_argument("--num_workers", type=int, default=40,
                    help="Number of workers to use while Gulping dataset.")

args = parser.parse_args()


def find_files(path, extension=".mp4"):
    """Iterates recursively through the directory provided and returns all files
       whose extension matches that provided.
    """
    all_files = []

    for root, subdirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                all_files.append(os.path.join(root, f))

    return sorted(all_files)


class CrossTaskGulpIO(AbstractDatasetAdapter, 
                      Custom20BNAdapterMixin):
    def __init__(self, dataset_path, annotations_path=None):
        frame_fnames = find_files(dataset_path, extension=".jpg")
        self.dataset, self.metadata = \
            self._build_dataset_dict(frame_fnames, annotations_path)

        self.dataset_keys = sorted(list(self.dataset.keys()))
        # self.labels2idx = self.create_label2idx_dict("videos")


    def _build_dataset_dict(self, frame_fnames, annotations_path=None):
        dataset = defaultdict(list)
        metadata = defaultdict(dict)
        
        for fname in frame_fnames:
            dirname = os.path.dirname(fname)

            youtube_id = os.path.basename(dirname)
            task_id = os.path.basename(os.path.dirname(dirname))

            if annotations_path is not None:
                annotations = os.path.join(annotations_path, 
                                        "{}_{}.csv".format(task_id, youtube_id))

                # Not all videos have annotations available.
                if os.path.exists(annotations):
                    annotations = open(annotations, 'r').readlines()
                    metadata[youtube_id]['annotations'] = annotations

            metadata[youtube_id]['task_id'] = task_id
            metadata[youtube_id]['youtube_id'] = youtube_id

            dataset[youtube_id].append(fname)

        return dataset, metadata


    def _read_frames(self, frame_fnames):
        for i, fname in enumerate(frame_fnames):
            yield io.imread(fname)


    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))

        for key in self.dataset_keys[slice_element]:
            id_ = key
            metadata = self.metadata[key]
            frames = list(self._read_frames(self.dataset[key]))

            yield {'id' : id_,
                   'meta' : metadata,
                   'frames' : frames}
        # else:
            # self.write_label2idx_dict()


    def __len__(self):
        return len(self.dataset_keys)


def main():
    adapter = CrossTaskGulpIO(args.dataset_path, args.annotations_path)

    ingestor = GulpIngestor(adapter, 
                            args.output_folder, 
                            args.videos_per_gulp, 
                            args.num_workers)

    ingestor()

    return 0


if __name__ == "__main__":
    sys.exit(main())
