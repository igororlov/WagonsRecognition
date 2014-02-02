#include "wagonsUtil.h"


Mat convertToGray(Mat& image) {
	Mat gray;

	if (image.channels() == 3) {
		cvtColor(image, gray, CV_BGR2GRAY);
	} else {
		gray = image;
	}
	return gray;
}

void concatMat(Mat first, Mat second, Mat& destination, bool vertical) {
	assert(destination.channels() == 1);
	
	int width = first.cols;
	int height = first.rows;

	Mat gray1 = convertToGray(first);
	Mat gray2 = convertToGray(second);

	gray1.copyTo(destination(Rect(0, 0, width, height)));
	if (vertical)
		gray2.copyTo(destination(Rect(0, height, width, height)));
	else // horizontal
		gray2.copyTo(destination(Rect(width, 0, width, height)));
}

int getBlurSize(int charHeight) {
	return charHeight / 8;
}