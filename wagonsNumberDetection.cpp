#include "wagonsNumberDetection.h"

using namespace cv;
using namespace std;

// ??? почему он внутри функции применяется, а сюда не возвращается???
// ссылка хреново работает!
Mat& sobelFilter(Mat &image)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	if ( image.channels() == 3 ) 
	{
		/// Convert it to gray
		cvtColor(image, image, CV_RGB2GRAY);
	}

  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  Sobel(image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

  /// Gradient Y
  Sobel(image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(grad_y, abs_grad_y);

  /// Total Gradient (approximate)
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image);

  return image;
}

/* scale - коэфф изменения размера изображения*/
vector<KeyPoint> detectCorners(Mat image, float scale, int fastThresh)  {
	vector<KeyPoint> keypoints;
	FastFeatureDetector fast(fastThresh);
	Mat resizedImage;

	resize(image,resizedImage,Size(image.cols/scale, image.rows/scale));
	fast.detect(resizedImage,keypoints);
	
	for (int j = 0; j < (int)keypoints.size(); j++) {
		// change keypoints coords * scale
		keypoints.at(j).pt.x *= scale;
		keypoints.at(j).pt.y *= scale;
	}
	return keypoints;
}

vector<vector<KeyPoint>> getKeypointGroups(vector<KeyPoint> keypoints, int RADIUS_X, int RADIUS_Y) {
	vector<vector<KeyPoint>> kp_groups;
	const int N = (int)keypoints.size();
	int i;
	int* id = new int[N]; // будет хранить "связные" области (как в алгоритме из книги)
	for (i = 0; i < N; i++) id[i] = i;

	// Перебор вершин
	for (int j = 0; j < N-1; j++) {
		for (int k = j+1; k < N; k++) {
			float X = keypoints.at(j).pt.x-keypoints.at(k).pt.x;
			float Y = keypoints.at(j).pt.y-keypoints.at(k).pt.y;
	
			if ( (int)(pow(X,2) / pow((float)RADIUS_X,2)) + 
				(int)(pow(Y,2) / pow((float)RADIUS_Y,2)) < 1 ) {
				int t = id[j];

				if (t == id[k]) continue;
				for (i = 0; i < N; i++)
				if (id[i] == t) id[i] = id[k];
			}
		}
	}

	vector<int> groups; 
	// будет хранить уникальные номера групп 
	// (Пр. 1 25 - то в id есть только все элементы равны или 1, или 25,
	// т.е. образуют две связные области)
	for (i = 0; i < N; i++) {   
		bool unique = true;
		for (int j = 0; j < groups.size(); j++) {
			if (id[i] == groups.at(j))
			unique = false;
		}
		if (unique) {
			groups.push_back(id[i]);
		}
	}

	//// для каждой группы создать вектор keypoint-ов, состоящий только их keypoint-ов этой группы
	for (int j = 0; j < groups.size(); j++) {
		vector<KeyPoint> tmp;
		// Проход по точкам
		for (i = 0; i < N; i++) {
			if (id[i] == groups.at(j)) {
				tmp.push_back(keypoints.at(i));
			}
		}
		kp_groups.push_back(tmp);
	}
	return kp_groups;
}