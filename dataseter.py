"""Create the dataset
"""

import random
from glob import glob

# loading random images from the video samples
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw
from tqdm import tqdm

from utils import *

vids = glob(os.path.expanduser("~/demos/vids/*"))

create_screenshots_from_video(video_path=vids[0])

# creating the train and test thing
import random
test_file  = open("datasets/humans/test.txt", "w+")
train_file= open("datasets/humans/train.txt", "w+")
test_size = 0.3

images = glob("datasets/humans/images/*")
test_files = random.sample(images, int(len(images)*test_size))
train_files = set(images) - set(test_files)

# removing the absolute paths from the images
train_files = [i.split("/")[-1].replace(".jpg", "") for i in train_files]
test_files = [i.split("/")[-1].replace(".jpg", "") for i in test_files]

strin = ""
for file in train_files:
    strin += file+"\n"
train_file.write(strin)
train_file.close()

strin = ""
for file in test_files:
    strin += file+"\n"
test_file.write(strin)
test_file.close()