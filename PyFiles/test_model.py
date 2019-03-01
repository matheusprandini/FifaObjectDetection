import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

# This is needed to display the images.
import matplotlib.pyplot as plt

from utils import label_map_util

from utils import visualization_utils as vis_util

path_inference_graph = '/home/mat/FifaObjectDetection/'

# What model to download.
MODEL_NAME = path_inference_graph + 'fifa_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = MODEL_NAME + 'object-detection.pbtxt'

PATH_TO_TEST_IMAGES_DIR = MODEL_NAME + "/test_images/"
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'fkrl{}.png'.format(i)) for i in range(131, 151) ]

print(TEST_IMAGE_PATHS)
