#-*- coding: utf-8 -*-
"""
Create on 2019/6/27

@Author: xhj
"""

import os
import cv2
import sys
import numpy as np
sys.path.append('../utils')
from utils import *


class PCB(object):
	
	def __init__(self, pcb_image):
		self.pcb_image = pcb_image