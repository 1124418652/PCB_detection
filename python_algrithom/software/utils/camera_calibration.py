#-*- coding: utf-8 -*-
"""
Created on 28 March

Author: xhj
"""
from __future__ import division
import os
import cv2
import glob
import logging
import numpy as np


# folder path of chessboard images
CHESSBOARD_IMAGE_PATH = os.path.join(os.path.abspath('.'), '..', 'chessboard_images')


# camera parameters for iphone6s plus

"""
相机内参矩阵
从相机坐标系到像素坐标系的变换矩阵
"""
INTRINSIC = np.array([[3.19506093e+03, 0.00000000e+00, 1.45576640e+03],
 			 		  [0.00000000e+00, 3.18840999e+03, 1.93038635e+03],
 			 		  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

"""
相机畸变参数
k1, k2, p1, p2, k3: 其中k1, k2, k3为相机径向畸变参数，p1, p2为相机切向畸变参数
径向畸变校正:
x_corrected = x(1 + k1r^2 + k2r^4 + k3r^6)
y_corrected = y(1 + k1r^2 + k2r^4 + k3r^6)
切向畸变校正:
x_corrected = x + [2p1y + p2(r^2 + 2x^2)]
y_corrected = y + [2p2y + p1(r^2 + 2y^2)]
"""
DISTORTION = np.array([[0.0581196, -0.19183175, 0.00280117, -0.00094675, 0.56864833]])

"""
相机外参，旋转向量
"""
ROTATE_VECTOR = [np.array([[ 0.0583518 ], [-0.19435355], [-0.05611541]]),
				 np.array([[-0.25059127], [-0.00976817], [ 2.98306391]]),
				 np.array([[-0.0153187 ], [ 0.10042951], [-3.07439429]]),
				 np.array([[ 0.06140688], [ 0.04547042], [-3.10007146]]),
				 np.array([[ 0.08474778], [-0.03596828], [-3.06570795]]),
				 np.array([[ 0.047351  ], [-0.08191684], [-2.97909085]]),
				 np.array([[-0.0239595 ], [-0.03947546], [ 3.06380572]]),
				 np.array([[-0.03258329], [-0.0579299 ], [ 3.13659523]]),
				 np.array([[-1.77628363e-04], [ 5.34128472e-02], [-3.04551002e+00]]),
				 np.array([[-0.05156838], [-0.0927781 ], [ 2.97959082]]),
				 np.array([[-0.04488124], [-0.1122059 ], [ 3.09641246]]),
				 np.array([[ 0.0793253 ], [ 0.09308744], [-3.12011426]])]

"""
相机外参，平移向量
"""
TRANSFER_VECTOR = [np.array([[-42.76412257], [-87.05105492], [255.25874521]]), 
				   np.array([[ 76.09350139], [ 84.73856548], [253.70922708]]),
				   np.array([[ 62.73024782], [ 96.7064068 ], [264.66465786]]), 
				   np.array([[ 69.91534851], [101.58227833], [247.42441455]]), 
				   np.array([[ 64.68372271], [ 94.90076824], [295.02896494]]), 
				   np.array([[ 57.10044891], [111.44049407], [273.55481563]]), 
				   np.array([[ 72.76313607], [ 90.32512252], [277.37422493]]), 
				   np.array([[ 72.17134885], [ 98.79621673], [275.38190614]]), 
				   np.array([[ 54.28242693], [103.83503003], [264.76998506]]), 
				   np.array([[ 87.45074321], [ 87.16392239], [260.61673871]]), 
				   np.array([[ 82.45651243], [ 96.85653666], [258.27412209]]), 
				   np.array([[ 67.60104008], [107.77415672], [260.64276629]])]


# camera parameters for huawei
INTRINSIC_H = np.array([[1.99373687e+04, 0.00000000e+00, 3.11752625e+03],
						[0.00000000e+00, 1.78368032e+04, 9.69097117e+02],
						[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

DISTORTION_H = np.array([-1.04030801e+00, 3.33099669e+01, -8.80089426e-05,
						 4.70328009e-03, -3.70133151e+02])
ROTATE_VECTOR_H = [np.array([[-0.07374469], [-0.53233845], [-0.03144597]])]
TRANSFER_VECTOR_H = [np.array([[-118.64783515], [  -2.22801881], [1005.60552746]])]


def get_chessboard_point_pairs(dir_path, grid_size = (6, 9), 
							   square_width = 30):
	"""
	find the location of chessboard corners in pixel image coordinate
	and world coordinate system. The corners will be placed in an order
	(from left-to-right, top-to-bottom)

	Parameters:
	dir_path: type of str, the folder path of chessboard image's files
	grid_size: type of tuple or list, (width, height)
	square_width: type of float, the width of every square in chessboard

	Returns:
	image_success: type of int, the number of images which find chessboard
				   corners successfully
	obj_points: 3d points list in real world space
	img_points: 2d points list in pixel image
	image_miss: type of list, store the filename of image which can't find 
				chessboard corners
	"""

	if not os.path.exists(dir_path):
		raise ValueError('The folder path does not exist!')

	# 指定亚像素迭代条件
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	# prepare object points, like (0,0,0),(1,0,0),(2,0,0)...(6,5,0)
	width, height = grid_size
	objp = np.zeros((width * height, 3), np.float32)
	objp[:, :2] = np.mgrid[0: width, 0:height].T.reshape((-1, 2)) * square_width

	# list to store object points and image points from all the images
	obj_points = []      # 3d points in real world space
	img_points = []      # 2d points in pixel images

	image_pathes = glob.glob(dir_path + '/*.*')
	image_miss = []
	image_success = 0    # 成功找到棋盘交点的图片数目
	for fname in image_pathes:
		try:
			img = cv2.imread(fname)
			print(img.shape)
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			# find the chessboard corners
			ret, corners = cv2.findChessboardCorners(gray_img, grid_size, None)
			if True == ret:   # 如果成功找到交点，则保存该幅图像的objp和corners
				obj_points.append(objp)
				cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
				img_points.append(corners)
				image_success += 1
				cv2.drawChessBoardCorners(img, grid_size, corners, ret)
				cv2.imshow('img', img)
				cv2.waitKey(500)
			else:
				image_miss.append(fname)

		except:
			image_miss.append(fname)
			continue

	return image_success, obj_points, img_points, image_miss


def camera_calibrate(dir_path, img_size, grid_size = (6, 9), square_width = 24,
					 flags = 0, criteria = None):
	"""
	find the camera intrinsic and extrinsic parameters from several views of a 
	calibration pattern.

	Parameters: 
	dir_path: type of str, the folder path of chessboard image's files
	img_size: type of list or tuple, the size of input images, (width, height)
	grid_size: type of tuple or list, the size of chessboard, (width, height)
	square_width: type of float, the width of every square in chessboard
	flags: flags used for function cv::calibrateCamera()
	criteria: Termination criteria for the iterative optimization algorithom
	"""

	image_success, obj_points, img_points, image_miss = get_chessboard_point_pairs(
		dir_path, grid_size, square_width)
	mtx = None   	# intrinsic matrix
	dist = None  	# list of distortion coefficients, of 4, 5, 8, 12 or 14 elements
	rvecs = None 	# rotation matrix
	tvecs = None 	# transform matrix

	# we need at least 10 test patterns for camera calibration
	print(image_success)
	if image_success < 1:
		ret = None
		return ret, mtx, dist, rvecs, tvecs

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, 
		img_size, flags, criteria)

	return ret, mtx, dist, rvecs, tvecs


def undistort_img(img, alpha = 0, camera_matrix = INTRINSIC, dist_coeffs = DISTORTION):
	"""
	Transforms an image to compensate for lens distortion
	
	Parameters:
	img: input (distorted image) image
	alpha: free scaling parameters between 0(all the pixel in the undistorted image are 
		   valid) and 1(all the source image pixels are retrained in the undistored image)
	camera_matrix: the intrinsic matrix of camera
	dist_coeffs: input vector of distortion coefficents
	new_camera_matrix: camera matrix of the distorted image. By default, it's the
					   same as camera intrinsic matrix but you may additionally scale
					   and shift the result by using a different matrix
	"""

	if not isinstance(img, np.ndarray):
		raise ValueError('The input image should be numpy.ndarray')

	height, width = img.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, 
		(width,height), 1, (width,height))
	dst = cv2.undistort(img, camera_matrix, dist_coeffs, None)
	x, y, w, h = roi 

	return dst[y: y+w, x:x+h]

# img = cv2.imread('../chessboard_images/pcb.jpg')
# # img = cv2.resize(img, (300, 400))
# cv2.namedWindow('img', 2)
# cv2.namedWindow('dst', 2)
# cv2.imshow('img', img)
# dst = undistort_img(img)
# cv2.imshow('dst', dst)

# cv2.waitKey(0)

img = cv2.imread('../pcb_images/pcb1.jpg')
height, width = img.shape[:2]
# ret, mtx, dist, rvecs, tvecs = camera_calibrate('../chessboard_images/', (width, height))

# print("ret", ret)
# print("mtx", mtx)
# print("dist", dist)
# print("rvecs", rvecs)
# print("tvecs", tvecs)

dst = undistort_img(img, 0, INTRINSIC, DISTORTION)
cv2.namedWindow('img', 2)
cv2.namedWindow('dst', 2)

cv2.imshow('img', img)
cv2.imshow('dst', dst)

cv2.waitKey(0)