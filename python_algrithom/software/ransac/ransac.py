#-*- coding: utf-8 -*-
"""
Create on 2019/9/14

@Author: xhj
"""

import cv2
import numpy as np 
from abc import ABC, abstractmethod


class Ransac(ABC):

	def __init__(self, samples=None):

		self.samples = samples
		self.random_seed = 0
		self.max_iterations = 100
		self.global_samples_num = 0
		if samples is not None:
			self.global_samples_num = len(samples)
		self.inlier_thresh = self.global_samples_num // 4


	def set_random_seed(self, random_seed):

		self.random_seed = random_seed
		np.random.seed(random_seed)


	def set_max_iterations(self, max_iterations=0):

		self.max_iterations = max_iterations


	@abstractmethod
	def computeModel(self, sample_num=3, stop_at_goal=True):
		pass
