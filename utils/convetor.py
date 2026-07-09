#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import datetime
import gc
import math
import os
import pickle
import re
import time
import atexit

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
# from torch.multiprocessing import Pool

from models.Nets import convert_lenet_weights
from models.test import Evaluator
from utils.attacker import Attacker
from utils.channelLipz import CL
from utils.defense import Defender
from utils.options import args_parser
from utils.dataUtils import getDataWithDistribution
from utils.sampling import plotDataDistribution
from utils.trainUtils import getModel, getWglob, getWglobTSSWeight, parallelTrainingIntegrated
from utils.trainUtils import train_user_attack, train_user_normal
from utils.assess import reverse_engineer
from utils.logger import myLogger


if __name__ == '__main__':
    model1 = getModel('lenetsm', 'mnist', input_size=28, device='cpu')
    model2 = getModel('lenet', 'mnist', input_size=28, device='cpu')

    w1 = model1.state_dict()
    w2 = model2.state_dict()

    w_converted = convert_lenet_weights(w1, 'softmask', 'standard')
    model2.load_state_dict(w_converted)

    w_converted = convert_lenet_weights(w2, 'standard', 'softmask')
    model1.load_state_dict(w_converted)
