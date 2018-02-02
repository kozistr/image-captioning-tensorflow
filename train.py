from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import random
import threading
import argparse as ap
from six.moves import xrange

import tensorflow as tf
import numpy as np


# Argument Parser
parser = ap.ArgumentParser()

parser.add_argument('dataset_dir',
                    default="D:/DataSet/coco_captioning/2017/", help="dataset directory")

parser.add_argument('pre-trained_model',
                    default='D:/DataSet/model/inception/inceptionv3-model.ckpt', help="inception-v3 pre-trained model")
parser.add_argument('model path',
                    default="D:/DataSet/model/coco/", help="saved model path")

parser.add_argument('batch_size',
                    default=128, help="batch_size")
parser.add_argument('global_steps',
                    default=1e6, help="training steps")
parser.add_argument('logging_steps',
                    default=1e3, help="logging steps")

parser.add_argument('num_threads',
                    default=8, help="the number of threads")
