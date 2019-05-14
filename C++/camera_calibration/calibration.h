#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#define CV
//#define FISHEYE

class CCalibration
{
public:
	CCalibration(std::string _patternImgPath, cv::Size _boardSize)
	{
		this->patternImgPath = _patternImgPath;
		this->boardSize = _boardSize;
	}
	CCalibration(std::string _calibResultPath):calibResultPath(_calibResultPath) {}
	~CCalibration() {};

private:
	std::vector<cv::Point3f> singlePatternPoints; // the corner points of all pattern images
	std::vector<cv::Mat> patternImgList;          // the list of pattern images
	int imgHeight = 0;
	int imgWidth = 0;
	int imgNum = 0;
	float scale = 0.25;
	float errThresh = 3000;
	std::string patternImgPath;           // the path of chessboard images' folder
	std::string calibResultPath;
	cv::Size boardSize = cv::Size(0, 0);  // the size of chessboard grid
	cv::Mat cameraMatrix;          // camera intrinsic matrix
	cv::Mat cameraDistortCoeff;    // camera distortion coefficients

private:
	int evaluateCalibrationResult(const std::vector<std::vector<cv::Point3f>> &objectPoints,
		const std::vector<std::vector<cv::Point2f>> &cornerSquare,
		const std::vector<int> &pointCnts,
		const std::vector<cv::Vec3d> &_rvec,
		const std::vector<cv::Vec3d> &_tvec, 
		const cv::Mat &_cameraMatrix,
		const cv::Mat &_cameraDistortCoeff,
		int count,
		const std::vector<int> &outLierIndex,
		float errThresh);
	bool testCorners(std::vector<cv::Point2f> &corners, int patternWidth, int patternHeight);
	void init3DPoints(cv::Size boardSize, cv::Size SquareSize,
		std::vector<cv::Point3f> &singlePatternPoint);

public:
	bool saveParams(std::string _calibResultPath);
	bool readPatternImg();
	void calibProcess();
	void run();
};