#include "main.h"

using namespace std;
using namespace cv;

static int VIDEO_TRACKBAR_VALUE = 0;
int frameNumber = 0;
VideoCapture capture;

void onTrackbarListener( int, void* )
{
	capture.set(CV_CAP_PROP_POS_FRAMES, VIDEO_TRACKBAR_VALUE);
}

Mat lpr(Mat &frame) {
	Mat gray;
	cvtColor(frame, gray, CV_BGR2GRAY);
	blur(gray, gray, Size(3, 3));
	
	//Find vertical lines. Car plates have high density of vertical lines
	Mat imgSobel;
	Sobel(gray, imgSobel, CV_8U, 1, 0, 3, 1, 0);

	//threshold image
	Mat imgThreshold;
	threshold(imgSobel, imgThreshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	
	// Morph close
	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(imgThreshold, imgThreshold, CV_MOP_CLOSE, element);

	return imgThreshold;
}

int main()
{
	int videoTrackNum = chooseVideoTrackNum();
	string videoTrackPath = getPathToVideo(videoTrackNum);
	if ( videoTrackPath.length() == 0 ) {
		return -1;
	}
	
	capture.open("C:\\Users\\Igor\\Dropbox\\WORK\\Video\\russia-day.avi");
	//capture.open("C:\\Main\\Work\\Video\\kazahstan.avi");
	//capture.open(videoTrackPath.c_str());

	int trackbarValue = 0;
	int totalFramesCount = (int)capture.get(CV_CAP_PROP_FRAME_COUNT); // общее к-во кадров на видео
	if (totalFramesCount <= 0) return 0;

	namedWindow("Main window");
	createTrackbar("Frame No", "Main window", &trackbarValue, totalFramesCount, onTrackbarListener);

	double rate = capture.get(CV_CAP_PROP_FPS); // frames per second
	int delay = 1000 / (int)rate;
	
	Mat frame;
	while (1) {
		if (!capture.read(frame))
			break;
		frame = lpr(frame);
		
		imshow("Main window", frame);
		char c = waitKey(delay);
		if (c == 27) {
			break;
		} else if (c == 32) {
			waitKey(0);
		}
		frame.release();
	}

	return 0;
}