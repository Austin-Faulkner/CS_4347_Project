from google.colab import drive

drive.mount('/content/drive', force_remount=True)

# install dependencies:
!pip install pyyaml==5.1

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version

# install detectron2
import torch
assert torch.__version__.startswith("1.7")
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

# Some basic setup:
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# This is the video we're going to process
from IPython.display import YouTubeVideo, display
video = YouTubeVideo("UnNSB0kmcwE", width=500)
display(video)

# Install dependencies, download the video, and crop 5 seconds for processing
!pip install youtube-dl
!pip uninstall -y opencv-python-headless opencv-contrib-python
!apt install python3-opencv  # the one pre-installed have some issues
!youtube-dl https://www.youtube.com/watch?v=UnNSB0kmcwE -f 22 -o video.mp4
!ffmpeg -i video.mp4 -t 00:00:36 -c:v copy video-clip.mp4

# Run frame-by-frame inference demo on this video (takes 3-4 minutes) with the "demo.py" tool we provided in the repo.
!git clone https://github.com/facebookresearch/detectron2
!python detectron2/demo/demo.py --config-file detectron2/configs/LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml --video-input video-clip.mp4 --confidence-threshold 0.6 --output video-output.mkv \
  --opts MODEL.WEIGHTS detectron2://LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x/144219035/model_final_824ab5.pkl

# Download the results
from google.colab import files
files.download('video-output.mkv')