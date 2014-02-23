#ifndef _WAGONS_UTIL_H
#define _WAGONS_UTIL_H

/*
 * Functions and declarations used for serving other parts,
 * like suporting functions for algorithms etc.
 */

#include <stdio.h>
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

using namespace cv;
using namespace std;

typedef enum {
	VERTICAL, HORIZONTAL
} Direction;

/*
 * Returns grayscale image from input 
 * (if RGB - converts and returns, if gray - returns image itself).
 */
Mat convertToGray(Mat& image);

/*
 * Concatenates two images into destination image in specified direction.
 * Note 1: Destination image must have pre-allocated memory
 * Note 2: Size of destination image must be sufficient to fit both images.
 */
void concatMat(Mat first, Mat second, Mat& destination, Direction concatDirection = HORIZONTAL);

/*
 * Returns size of square side for blurring the original frame.
 */
int getBlurSize(int charHeight);

#endif