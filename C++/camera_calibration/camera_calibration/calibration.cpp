#include "stdafx.h"
#include "calibration.h"

bool CCalibration::saveParams(std::string _calibResultPath)
{
	if (!cameraMatrix.data || !cameraDistortCoeff.data)
		return false;
	assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);
	cameraMatrix.convertTo(cameraMatrix, CV_32FC1);
	cameraDistortCoeff.convertTo(cameraDistortCoeff, CV_32FC1);
	std::ofstream fout;
	fout.open(_calibResultPath + "calibResult.txt", std::ios::out);
}