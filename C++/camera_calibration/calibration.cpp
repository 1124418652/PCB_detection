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
	for (int i = 0; i < 3; ++i)
	{
		float *pRow = cameraMatrix.ptr<float>(i);
		for (int j = 0; j < 3; ++j)
		{
			fout << *(pRow++) << '\t';
		}
		fout << std::endl;
	}

#ifdef CV
	fout << "Distortion coefficients:" << std::endl;
	fout << cameraDistortCoeff.at<float>(0, 0) << '\t';
	fout << cameraDistortCoeff.at<float>(0, 1) << '\t';
	fout << cameraDistortCoeff.at<float>(0, 2) << '\t';
	fout << cameraDistortCoeff.at<float>(0, 3) << '\t';
	fout << cameraDistortCoeff.at<float>(0, 4) << '\t';
#elif define FISHEYE
	fout << "Distortion coefficients with fisheye camera:" << endl;
	fout << cameraDistortCoeff.at<float>(0, 0) << '\t';
	fout << cameraDistortCoeff.at<float>(0, 1) << '\t';
	fout << cameraDistortCoeff.at<float>(0, 2) << '\t';
	fout << cameraDistortCoeff.at<float>(0, 3) << '\t';
#endif
	fout.close();
	return true;
}

bool CCalibration::readPatternImg()
{
	cv::Mat img;
	std::vector<std::string> files;
	
}