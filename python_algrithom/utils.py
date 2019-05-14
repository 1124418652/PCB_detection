#-*- coding: utf-8 -*-
"""
Create on 2019/5/14

@Author: xhj
"""

import cv2
import numpy as np
from collections import deque


def find_circles(image, mask=None, threshold=200, nmax=100,
				 rmin=5, rmax=10, rstd=1, rpan=4):
	"""
	基于hough梯度变换在图片中找到圆形所在的位置信息

	Args:
		image: 输入图片，BGR格式的彩色图
		mask:
		threshold: int类型，表示筛选circles时score的最小值
		nmax: int类型，表示筛选出权重最高的前nmax个circles
		rmin:
		rmax:
		rstd:
		rpan:
	"""

	if rmin < rpan + 1:
		raise ValueError

	if mask is not None:
		rmin = max(rmin, int(np.ceil(np.min(mask) - rstd)))
		rmax = min(rmax, int(np.floor(np.max(mask) + rstd)))

	if rmin > rmax:
		return [], []

	# 基于灰度图产生梯度信息
	Dx = cv2.Scharr(image, cv2.CV_32F, 1, 0)
	Dy = cv2.Scharr(image, cv2.CV_32F, 0, 1)
	Da = np.arctan2(Dy, Dx) * 2       # np.arctan2 的取值范围是 -pi ~ pi
	Ds = np.log1p(np.hypot(Dy, Dx))   # = log(1+sqrt(Dy^2+Dx^2))
	Du = np.sum(np.cos(Da) * Ds, axis=-1)  # 将 BGR 三个通道的值进行合并
	Dv = np.sum(np.sin(Da) * Ds, axis=-1)

	# calculate likelihood for each (x, y, r) pair
	# based on: gradient changes across circle
	def iter_scores():
		queue = deque()
		for radius in range(rmin - rpan, rmax + rpan + 1):
			r = int(np.ceil(radius + 6 + rstd * 4))
			Ky, Kx = np.mgrid[-r: r+1, -r: r+1]
			Ka = np.arctan2(Ky, Kx) * 2
			Ks = np.exp(np.square(np.hypot(Ky, Kx) - radius) / 
						(-2 * rstd**2)) / np.sqrt(radius)
			Ku = np.cos(Ka) * Ks 
			Kv = np.sin(Ka) * Ks 
			queue.append(cv2.filter2D(Du, cv2.CV_32F, Ku) + 
						 cv2.filter2D(Dv, cv2.CV_32F, Kv))
			if len(queue) > rpan * 2:
				yield (radius - rpan, queue[rpan] - 
					   (np.fmax(0, queue[0]) + np.fmax(0, queue[rpan*2])))
				queue.popleft()

	# choose best (x, y, r) for each (x, y)
	radiuses = np.zeros(image.shape[:2], dtype=int)
	scores = np.full(image.shape[:2], -np.inf)
	for radius, score in iter_scores():
		sel = (score > scores)
		if mask is not None:
			sel &= (mask > radius - rstd) & (mask < radius + rstd)
		scores[sel] = score[sel]
		radiuses[sel] = radius

	# choose the top n circles
	circles = []
	weights = []
	for _ in range(nmax):
		# 在每一步循环中找到 scores 矩阵中的最大值的坐标，np.argmax 返回的是
		# 数组拉平之后的最大值下标，需要使用 np.unravel_index 转换为原数组中
		# 的最大值的下标，y是行下标，x是列下标
		y, x = np.unravel_index(np.argmax(scores), scores.shape)
		score = scores[y, x]
		if score < threshold:     # score是当前最大值，如果该值小于阈值，说明没必要继续后面的循环
			break
		r = radiuses[y, x]
		circles.append((x, y, r))
		weights.append(score)
		cv2.circle(scores, (x, y), r, 0, -1)
	return circles, weights

def extract_sift_features(image, drawKP = False):
	"""
	使用opencv提供的sift算法进行关键点提取和sift描述子向量提取。

	Args:
		image: 进行检测的图片
		drawKP: boolean类型，表示是否需要在图片中绘制关键点
	Returns:
		key_points: 检测得到的关键点
		descriptor: 描述子向量
	"""

	try:
		if image.ndim > 2:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	except:
		raise ValueError

	sift = cv2.xfeatures2d.SIFT_create()
	key_points, descriptor = sift.detectAndCompute(image, None)
	if drawKP:
		image = cv2.drawKeypoints(image, key_points, image, color=(255, 0, 255))
	return key_points, descriptor, image


def extract_pcb_ragion(image, limit1=50, limit2=100, threshold1=0.5,
						threshold2=0.3):
	"""
	通过HSV格式图片中的H值范围，进行图片中PCB区域的提取，PCB表面有绿油覆盖，
	所以其H通道的值集中在绿色区域

	Args: 
		image: 输入图片，BGR格式
		limit1: H通道的区间下限
		limit2: H通道的区间上限
		threshold1: 行阈值比例
		threshold2: 列阈值比例
	Return:
		pcb_img: _PCB类型的对象
	"""

	if not 3 == image.ndim:
		raise ValueError("The input image must be BGR format")

	class _PCB():
		def __init__(self, row_begin, row_end, col_begin, col_end):
			self.row_begin = row_begin
			self.row_end = row_end
			self.col_begin = col_begin
			self.col_end = col_end
		def __str__(self):
			return "PCB roi: \nleft-top : (%(left)s, %(top)s)\nright-bottom: (%(right)s, %(bottom)s)"\
					% {"left": self.col_begin, "top": self.row_begin,
					   "right": self.col_end, "bottom": self.row_end}
		def roi(self):
			if self.row_begin < self.row_end and self.col_begin < self.col_end:
				return image[self.row_begin: self.row_end, self.col_begin: self.col_end, :]
			else:
				return None

	height, width = image.shape[:2]
	row_thresh, col_thresh = width * threshold1, height * threshold2
	hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	h_img, s_img, v_img = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]

	import matplotlib.pyplot as plt 
	h_row = []    # height维的数组，用于记录每一行的区域内像素个数
	h_col = []    # width维的数组，用于记录每一列的区域内像素个数
	for row in h_img:
		count = ((row >= limit1) & (row <= limit2)).sum()
		h_row.append(count)
	for col in range(width):
		count = ((h_img[:, col] >= limit1) & (h_img[:, col] <= limit2)).sum()
		h_col.append(count)

	row_begin = height - 1
	row_end = 0
	for index, row_count in enumerate(h_row):
		if row_count > row_thresh:
			row_begin = index 
			for index_inv, row_count_inv in enumerate(h_row[::-1]):
				if row_count_inv > row_thresh:
					row_end = height - index_inv
					break
			break

	col_begin = width - 1
	col_end = 0
	for index, col_count in enumerate(h_col):
		if col_count > col_thresh:
			col_begin = index 
			for index_inv, col_count_inv in enumerate(h_col[::-1]):
				if col_count_inv > col_thresh:
					col_end = width - index_inv
					break
			break

	return _PCB(row_begin, row_end, col_begin, col_end)


def rotate_pcb_image(image):
	if not 3 == image.ndim:
		raise ValueError

	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_img = cv2.bilateralFilter(gray_img, 10, 50, 30)#.astype(np.int32)
	# gray_img = 1 / (1 + np.exp(-gray_img + 120)) * 255
	# # gray_img = gray_img - 100
	# gray_img = np.where(gray_img > 0, gray_img, 0)
	kernel = np.ones((10, 10), np.uint8)
	gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
	gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_DILATE, kernel)
	gray_img = gray_img.astype(np.uint8)
	cv2.imshow('gray', gray_img)


if __name__ == '__main__':
	image_path = '../pcb_images/pcb_inv1.jpg'
	image = cv2.imread(image_path, 1)
	image = cv2.resize(image, (800, 600))

	# circles, weights = find_circles(image)
	# for (x, y, r) in circles:
	# 	cv2.circle(image, (x, y), r, 0, -1)

	# key_points, descriptor, image = extract_sift_features(image)
	# print(dir(key_points[0]))

	# cv2.imshow('image', image)
	# cv2.waitKey(0)

	pcb = extract_pcb_ragion(image)
	rotate_pcb_image(image)
	# gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# thresh_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)[1]
	# cv2.imshow('gray_img', thresh_img)
	# cv2.imshow('pcb', pcb.roi())
	cv2.waitKey(0)