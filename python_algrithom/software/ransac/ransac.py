#-*- coding: utf-8 -*-
"""
Create on 2019/9/14

@Author: xhj
"""

import cv2
import numpy as np 
from abc import ABC, abstractmethod


class Ransac(ABC):

	def __init__(self, samples=None, distance_thresh=.1):
		"""
		constructor of Ransac
		Args:
			samples: numpy.ndarray, the total samples
			distance_thresh: float, the threshold of distance between inliers 
							 to the model computed
		"""

		self.samples = samples
		self.random_seed = 0
		self.max_iterations = 100
		self.global_samples_num = 0
		if samples is not None:
			self.global_samples_num = len(samples)
		self.inlier_thresh = self.global_samples_num // 4
		self.distance_thresh = distance_thresh


	def set_random_seed(self, random_seed):

		self.random_seed = random_seed
		np.random.seed(random_seed)


	def set_max_iterations(self, max_iterations=0):

		self.max_iterations = max_iterations


	def set_distance_thresh(self, distance_thresh):

		self.distance_thresh = distance_thresh


	@abstractmethod
	def computeModel(self, sample_num=3, stop_at_goal=True):
		pass
