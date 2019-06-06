#-*- coding: utf-8 -*-
"""
Created on 2019/6/6

@Author: xhj
"""

import os
import sys
import cv2
import time
import numpy as np


def image_taking(save_dir=None, time_decay=500):
	"""
	负责图片的采集工作
	Args:
		save_dir: 存储图片的文件夹
		time_decay: 每一帧的延时
	"""

	quit = False
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	capture = cv2.VideoCapture(0)
	if not capture.isOpened():
		raise ValueError
	size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
			int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	
	flag, frame = capture.read()
	while flag and not quit:
		key = cv2.waitKey(time_decay)
		if key == ord('s'):
			image_file = os.path.join(save_dir,str(int(time.time()))+'.jpg')
			print("saving image to: ", image_file)
			print("size of image: ", frame.shape)
			cv2.imwrite(image_file, frame)
			print('finishing saving')
		elif key == ord('q') or key == 27:
			quit = True
		cv2.imshow("Video", frame)
		flag, frame = capture.read()





if __name__ == '__main__':
	image_taking('../source_image')
	