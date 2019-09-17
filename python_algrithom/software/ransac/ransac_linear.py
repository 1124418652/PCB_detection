#-*- coding: utf-8 -*-
"""
Create on 2019/9/14

@Author: xhj
"""


import time
import numpy as np
from ransac import Ransac


class RansacLinear(Ransac):

	def __init__(self, samples=None):

		Ransac.__init__(self, samples)


	def computeModel(self, sample_num=3, stop_at_goal=True):

		max_inliers_num = 0
		best_model = None
		skip_sample_num = 0
		idx = 0
		max_iterations = self.max_iterations
		random_sample_array = list(np.random.randint(0, self.global_samples_num, \
													 (self.max_iterations, sample_num)))

		while idx < max_iterations:
			
			sample_index = random_sample_array[idx]
			if len(set(sample_index)) != len(sample_index):
				random_sample_array.append(np.random.randint(0, self.global_samples_num, sample_num))
				max_iterations += 1
				skip_sample_num += 1
				if skip_sample_num > self.global_samples_num / 10:
					return
				continue

			coeff_calc = lambda samples: np.linalg.svd(samples)[-1][-1, :]
			coeff = coeff_calc(self.samples[sample_index])

			idx += 1

if __name__ == '__main__':

	samples = np.random.randint(0, 100, (1000, 3))
	ransac = RansacLinear(samples)
	ransac.computeModel()
