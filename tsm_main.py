import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


import torch
import torch.hub

from gulpio.loader import DataLoader
from gulpio.transforms import Scale, ComposeVideo 

from dataset import VideoDataset


dataset_path = "./data"

repo = "epic-kitchens/action-models"

class_counts = (125, 352)
segment_count = 8
base_model = "resnet50"

batch_size = 1
segment_count = 8
snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
height, width = 224, 224

scale = ComposeVideo([Scale((height, width))])

# Eight segments each composed of one frame.
# Each segment is ten seconds apart.
dataset = video_dataset = VideoDataset(dataset_path, 
                                       num_frames=snippet_length * segment_count, 
                                       step_size=10, 
                                       transform=scale, 
                                       is_val=False)

loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

tsm = torch.hub.load(repo, "TSM", class_counts, segment_count, "RGB",
                     base_model=base_model, 
                     pretrained="epic-kitchens")


for data in loader:
    data = data.transpose((0, 1, 4, 2, 3))
    data = torch.FloatTensor(data)
    data = data.reshape((batch_size, -1, height, width))

    features = tsm.features(data)
    verb_logits, noun_logits = tsm.logits(features)
    print(verb_logits.shape, noun_logits.shape)
