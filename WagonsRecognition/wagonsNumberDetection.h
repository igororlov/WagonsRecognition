#ifndef _WAGONS_NUMBERDETECTION_H_
#define _WAGONS_NUMBERDETECTION_H_

#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\features2d\features2d.hpp"

#include "wagonsDrawing.h"
#include "wagonsPreferences.h"
#include "wagonsUtil.h"

using namespace cv;
using namespace std;

typedef enum {
	DETECTION_MORPHOLOGY,
	DETECTION_CORNERS
} detectionAlgorithm;

Mat& sobelFilter(Mat &image);
vector<KeyPoint> detectCorners(Mat image, float scale, int fastThresh=25);
vector<vector<KeyPoint>> getKeypointGroups(vector<KeyPoint> keypoints, int RADIUS_X, int RADIUS_Y);

Mat detection(Mat &frame);

class Detector {
public:
	virtual void detect(const Mat &inputImage) = 0;
	vector<Rect> getRects();
	void drawRects(Mat &image);
protected:
	bool verifySize(Rect rect);
	vector<Rect> _rects;
};

class MorphologyDetector : public Detector {
public:
	virtual void detect(const Mat &inputImage);
	void morphDetect(const Mat &inputImage, Mat& outputImage);
};

// TODO add corners detector

#endif