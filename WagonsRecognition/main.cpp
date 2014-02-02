#include "main.h"

using namespace std;
using namespace cv;

static int SLIDER_POSITION = 0;
VideoCapture capture;

void onTrackbarListener(int pos, void*) {
	if (WITH_TRACKBAR) {
		capture.set(CV_CAP_PROP_POS_FRAMES, SLIDER_POSITION);
	}
}

int main() {
	int videoTrackNum = chooseVideoTrackNum();
	string videoTrackPath = getPathToVideo(videoTrackNum);
	if ( videoTrackPath.length() == 0 ) {
		return -1;
	}
	
	capture.open(videoTrackPath.c_str());

	int totalFramesCount = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
	assert(totalFramesCount > 0);

	namedWindow(WINDOW_NAME);
	createTrackbar(TRACKBAR_NAME, WINDOW_NAME, &SLIDER_POSITION, totalFramesCount, onTrackbarListener);

	double rate = capture.get(CV_CAP_PROP_FPS); // frames per second
	int delay = 1000 / (int)rate;
	
	Mat frame;
	while (1) {
		if (!capture.read(frame))
			break;

		if (WITH_TRACKBAR)
			setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, ++SLIDER_POSITION);
		
		Mat detected = detection(frame);

		Mat target(frame.rows, frame.cols*2, CV_8UC1);
		concatMat(frame, detected, target, false);
		imshow(WINDOW_NAME, target);
		
		char c = waitKey(delay);
		if (c == 27) {
			break;
		} else if (c == 32) {
			waitKey(0);
		}
		frame.release();
	}
	capture.release();
	return 0;
}