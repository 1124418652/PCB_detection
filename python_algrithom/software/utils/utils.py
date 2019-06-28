#-*- coding: utf-8 -*-
"""
Create on 2019/5/14

@Author: xhj
"""

import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def extract_pcb_region(image):
	"""
	从原图中提取出只包含PCB图像的区域
	Args:
		image: BGR格式的输入图片
	"""

	if not isinstance(image, np.ndarray) or 3 != image.ndim:
		return False

	height, width = image.shape[:2]

	class PCB_region():

		@property
		def rect(self):
			return self._rect

		@rect.setter
		def rect(self, rect):
			if not 4 == len(rect):
				raise ValueError
			self._rect = rect

		@property
		def pcb_image(self):
			return self._pcb_image
		
		@pcb_image.setter
		def pcb_image(self, pcb_image):
			if not (isinstance(pcb_image, np.ndarray) and 3 == pcb_image.ndim):
				raise ValueError
			self._pcb_image = pcb_image

		@property
		def rotate_matrix(self):
			return self._rotate_matrix

		@rotate_matrix.setter
		def rotate_matrix(self, rotate_matrix):
			self._rotate_matrix = rotate_matrix
		
	hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	s_channel = hsv_img[..., 1]
	bilater_image = cv2.bilateralFilter(s_channel, 10, 20, 30)
	thresh, thresh_image = cv2.threshold(bilater_image, 0, 255, cv2.THRESH_OTSU)
	kernel = np.ones((15, 15), dtype=np.uint8)
	thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

	# 提取PCB图片的最小外接矩形
	_, contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, 
		cv2.CHAIN_APPROX_SIMPLE)
	contour_image = np.zeros((height, width, 3), np.uint8)
	max_contour_area = height * width / 4
	_rect = []
	for index, contour in enumerate(contours):
		tmp_contour_area = cv2.contourArea(contour)
		if tmp_contour_area > max_contour_area:
			contour_poly = cv2.approxPolyDP(contour, epsilon=5., closed=True)
			_rect.insert(0, cv2.minAreaRect(contour_poly))
			max_contour_area = tmp_contour_area

	if not len(_rect):
		return False

	# 旋转调整
	points = cv2.boxPoints(_rect[0]) 
	angle = _rect[0][2]
	if abs(angle) >= 45:         # PCB摆放的倾角不能大于45度
		rotate_angle = 90 + angle
	else:
		rotate_angle = angle
	rotate_matrix = cv2.getRotationMatrix2D(_rect[0][0], rotate_angle, scale=1)
	rotate_image = cv2.warpAffine(image, rotate_matrix, image.shape[1::-1])
	_points = np.transpose(np.concatenate([points, np.ones((4, 1))], axis=-1))
	_points = np.transpose(np.int0(np.matmul(rotate_matrix, _points)))

	# 截取PCB区域
	x_min, y_min = np.min(_points, axis=0)[:]
	x_max, y_max = np.max(_points, axis=0)[:]
	pcb_image = rotate_image[y_min: y_max, x_min: x_max, :]

	_pcb = PCB_region()

	try:
		_pcb.rect = points
		_pcb.rotate_matrix = rotate_matrix
		_pcb.pcb_image = pcb_image
	except:
		return False

	return _pcb


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


def find_position_circles(pcb_image, w_proportion=0.05, threshold=200, 
	nmax=10, rmin=5, rmax=10):
	"""
	从截取出的PCB图片中提取四个定位圆的中心坐标
	Args:
		pcb_image: np.ndarray 类型，BGR格式的图片
		w_proportion: float type, 表示定位圆所在位置的比例
	"""
	
	if not isinstance(pcb_image, np.ndarray):
		return False

	class Tmp_region():

		def __init__(self, region, x_begin, y_begin):
			self.region = region
			self.x_begin = x_begin
			self.y_begin = y_begin

	height, width = pcb_image.shape[:2]
	region_w = int(width * w_proportion)
	tmp_regions = []
	center_points = []
	tmp_regions.append(Tmp_region(pcb_image[:region_w, :region_w, :], 0, 0))
	tmp_regions.append(Tmp_region(pcb_image[:region_w, -region_w:, :], 
								  width-region_w, 0))
	tmp_regions.append(Tmp_region(pcb_image[-region_w:, :region_w, :], 
								  0, height-region_w))
	tmp_regions.append(Tmp_region(pcb_image[-region_w:, -region_w:, :], 
								  width-region_w, height-region_w))
	
	for region in tmp_regions:
		circles, weights = find_circles(region.region, threshold=threshold, 
			nmax=nmax, rmin=rmin, rmax=rmax)
		if len(weights) > 0:
			score_max = weights[0]
			index_max = 0
			for index, score in enumerate(weights): # find the circle with highest score
				if score > score_max:
					score_max = score
					index_max = index
			center_points.append((circles[index_max][0]+region.x_begin,
								  circles[index_max][1]+region.y_begin))

	if 4 > len(center_points):
		return False

	return center_points


def get_perspective_result(source_points, dest_points, pcb_image):
	"""
	获取透视变换的变换矩阵以及经过透视变换之后的PCB图片
	Args:
		source_points: 包含至少4个坐标点的列表
		dest_points: 与source_points对应的坐标点列表
		pcb_image: 待调整的PCB图片
	"""

	if len(source_points) < 4 or len(source_points) != len(dest_points):
		raise ValueError
	source_points = np.float32([source_points])
	dest_points = np.float32([dest_points])
	M = cv2.getPerspectiveTransform(source_points, dest_points)
	Minv = cv2.getPerspectiveTransform(dest_points, source_points)
	warped_image = cv2.warpPerspective(pcb_image, M, pcb_image.shape[1::-1],
									   flags=cv2.INTER_LINEAR)
	return warped_image, M, Minv


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
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	except:
		raise ValueError

	sift = cv2.xfeatures2d.SIFT_create()
	key_points, descriptor = sift.detectAndCompute(image_gray, None)
	if drawKP:
		image = cv2.drawKeypoints(image, key_points, image, color=(0, 0, 255))
	return key_points, descriptor, image


if __name__ == '__main__':
	image = cv2.imread("../../source_image/1560820953.jpg")
	pcb = extract_pcb_region(image)
	pcb_image = pcb.pcb_image
	height, width = pcb_image.shape[:2]
	source_points = find_position_circles(pcb_image)
	# print(source_points)
	# print(pcb_image.shape)
	# for point in source_points:
	# 	print(1)
	# 	cv2.circle(pcb_image, point, 8, (0, 0, 255), 2)

	dest_points = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
	pcb_image, *_ = get_perspective_result(source_points, dest_points, pcb_image)
	cv2.imwrite("../../source_image/warpedImage.jpg", pcb_image)
	cv2.imshow("pcb image", pcb_image)
	cv2.waitKey(0)