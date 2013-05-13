#include "main.h"

using namespace std;
using namespace cv;

int g_slider_position = 0;
CvCapture* videoCapture = NULL;

void onTrackbarSlide(int pos) {
	cvSetCaptureProperty(videoCapture, CV_CAP_PROP_POS_FRAMES, pos);
}

int main()
{
	int videoTrackNum = chooseVideoTrackNum();
	string path = getPathToVideo(videoTrackNum);
	if ( path.length() == 0 ) {
		return -1;
	}
	
	cvNamedWindow("Main window", CV_WINDOW_AUTOSIZE);
	videoCapture = cvCreateFileCapture(path.c_str());
	
	int framesCount = (int) cvGetCaptureProperty(videoCapture, CV_CAP_PROP_FRAME_COUNT);
	if( framesCount != 0 ) {
		cvCreateTrackbar("FrameNo", "Main window", &g_slider_position, framesCount, onTrackbarSlide);
	}
	
	IplImage* frame;
	while(1) {
		frame = cvQueryFrame(videoCapture);
		if ( !frame ) { 
			break;
		}
		setTrackbarPos("FrameNo", "Main window", 1 + getTrackbarPos("FrameNo", "Main window"));
		
		Mat image(frame);
		
		vector<KeyPoint> keypoints = detectCorners(image, 0.5, 15);
		drawKeypoints(image, keypoints, image, COLOR_GREEN_CPP, DrawMatchesFlags::DRAW_OVER_OUTIMG);
		vector<vector<KeyPoint>> kpGroups = getKeypointGroups(keypoints, 20, 10);
		for ( int i = 0; i < kpGroups.size(); i++) {
			vector<Point> points;
			for ( int j = 0; j < kpGroups.at(i).size(); j++ ) {
				Point p = kpGroups.at(i).at(j).pt;
				points.push_back(p);
			}
			Rect rect = boundingRect(points);
			rectangle(image, rect, COLOR_RED_CPP);
		}

		cvShowImage("Main window", frame);
		
		char c = cvWaitKey(33);
		if ( c == 27 ) {
			break;
		} else if ( c == 32 ) {
			cvWaitKey(0);
		}
	}
	cvReleaseCapture(&videoCapture);
	cvDestroyWindow("Main window");

	return 0;
}