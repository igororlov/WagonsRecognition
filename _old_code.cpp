////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

// OPENCV includes BEGIN
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\video\tracking.hpp>
#include <opencv2\stitching\stitcher.hpp>
// OPENCV includes END

#include <math.h>
#include <iostream>

using namespace cv;

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

// Глобальные переменные (использующиеся несколько раз в разных кусках, чтобы не было проблем с их хранением и передачей)
int NUMBER_HEIGHT = 30;
int SE_WIDTH = 40;
int thresh = NUMBER_HEIGHT*4*255; // порог для отбрасывания секторов средняя ширина номера = высота / 2, номер состоит из 4-х символов 
int NUM_SECTORS = 0; 
int* histogram;
int* minPositions; // локальные минимумы гистограммы
int NUM_MINS;
int CORNERS_THRESH = 15; // порог при поиске уголков, должен меняться в зависимости от размера номера
float COMPR_RATE = 2.; // коэффициент сжатия для поиска уголков (типа пирамиды)

int RADIUS_X = 25; // отдельно по осям Х
int RADIUS_Y = 10; // и Y в эллипсе

int VIDEO_TRACKBAR_MAX_VALUE = 100;
int VIDEO_TRACKBAR_VALUE = 0;
int RADIUS_TRACKBAR_MAX_VALUE = 50;
int RADIUS_X_TRACKBAR_VALUE = RADIUS_X;
int RADIUS_Y_TRACKBAR_VALUE = RADIUS_Y;
int COMPR_RATE_TRACKBAR_VALUE = (int)COMPR_RATE;
int COMPR_RATE_TRACKBAR_MAX_VALUE = 5;
int FASTTHRESH_TRACKBAR_VALUE = CORNERS_THRESH;
int FASTTHRESH_TRACKBAR_MAX_VALUE = 50;

int UP_BORDER = 0;
int DOWN_BORDER = 800;
int borders_setting = 0; // 0 - не включено, 1 - ждем первую границу, 2 - ждем вторую.

int HEIGHT_THRESH = 5; // пороги для трекинга - оставлять точки, 
int WIDTH_THRESH = 25; // которые на новом кадре передвинулись не более чем данные пороги.

float ANGLE_THRESH = 10.0; // оставлять только BBox с углами наклона из промежутка (-ANGLE_THRESH,ANGLE_THRESH)
float h_alpha_up = 1.9; // насколько ВВЕРХ допустимы отклонения по высоте для minBBox'а (если 1.0 = NUMBER_HEIGHT)
float h_alpha_down = 0.85; // и ВНИЗ
int MIN_BBOX_AREA = NUMBER_HEIGHT * NUMBER_HEIGHT;

VideoCapture capture;
int num_frame = 0;
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

// Объявления классов и функций.
class Region;

// Простые функции
char* getVideoTrack();
cv::Mat binarize(cv::Mat image);
float whiteRatio(cv::Mat image);
int* medFiltHist(int* arr,int n);
int* calcHist(cv::Mat image,int flag=0); // flag = 0 - hor, = 1 - vert
int* min(int* arr,int n);
int* max(int* arr,int n);
void printArray(int* arr,int n);
void prewittFilter(const cv::Mat &inp_image,cv::Mat &out_image,int type=1);
cv::Mat makeBlack(cv::Mat image);
int mean(int* arr,int n);
int* cleanHist(int* inp_arr,int n, int thresh);
bool belongs(int value, int* arr,int n);
int* mysort(int* arr,int n);
bool belongsToTriangle(cv::Point2f point,std::vector<cv::Point2f> vertices);
bool belongsToArea(cv::Point2f point,std::vector<cv::Point2f> vertices);
cv::Mat invertImage(cv::Mat image);

// Listener-ы для изменения настроек
void on_trackbar_listener( int, void* );
void on_trackbar2_listener( int, void* );
void on_trackbar3_listener( int, void* );
void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ );

// Вспомогательные функции
int* copyPartOfArray(int* inp_arr,int start,int finish);
int* localMins(int* arr,int n,int win_width = 20);
cv::Mat drawHist(cv::Mat image);
void imrotate(cv::Mat image,cv::Mat &out_image,double angle);
void drawBorders(cv::Mat &image);
void tmpShow(cv::Mat image);
float calcMatMorphoDensity(cv::Mat img);
bool checkRectOverlap(cv::Rect rect1,cv::Rect rect2);
Region joinRegions(Region region1,Region region2,cv::Mat frame);
Rect widenRect(Rect rect,int pixels = 2);
cv::Mat concatImages(cv::Mat img1,cv::Mat img2);
int getBinImgHeight(Mat charImage);
int getBinImgWidth(Mat charImage);
void printIntVector(std::vector<int> vect);
void drawLocalMins(Mat &image,std::vector<int> local_mins,cv::Scalar color = cv::Scalar(128,128,128));
int* getVertHist(Mat binImage);
void widenFrame(Mat &frame,int pixels);
cv::Mat cutCharWhite(cv::Mat charImage,cv::Mat binImage);

// Временно неиспользуемые
std::vector<cv::RotatedRect> makeTracking(std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts,cv::Mat &curr_image,cv::Mat prev_image,std::vector<cv::RotatedRect> prev_boxes);
std::vector<cv::DMatch> getBriskMatches(cv::Mat curr_frame, cv::Mat prev_frame, std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts);
cv::Mat drawBriskMatches(cv::Mat curr_frame, cv::Mat prev_frame, std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts,std::vector<cv::DMatch> matches);
std::vector<cv::DMatch> cleanMatches(std::vector<cv::DMatch> matches,std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts,int thresh);
cv::Mat drawLinesHoughDetection(cv::Mat image);
cv::Mat getContours(cv::Mat image);


// Важные функции
std::vector<cv::KeyPoint> detectCorners(cv::Mat image, float ratio ,int fast_thresh=25) ;
std::vector<std::vector<cv::KeyPoint>> getKeypointGroups(std::vector<cv::KeyPoint> keypoints,int RADIUS_X,int RADIUS_Y);
void getTrueBoxWidthHeight(RotatedRect bBox,float &width,float &height);

cv::RotatedRect getMinBBox(cv::Mat image,std::vector<cv::Point> points);
cv::Mat drawMinBBox(cv::Mat image, cv::RotatedRect box,cv::Scalar color);
bool checkedMinBBox(cv::RotatedRect box);
void clearKeypointsByAreaHeight(std::vector<cv::KeyPoint> keypoints,std::vector<cv::KeyPoint> &out_keypoints);
cv::Mat getBBoxImage(cv::Mat image,cv::RotatedRect box);

std::vector<Region> getJoinedRegions(std::vector<Region> regions,cv::Mat frame);
std::vector<cv::Mat> getUnitedImages(std::vector<Region> regions,cv::Mat frame);
std::vector<cv::KeyPoint> cleanStaticKeypts(std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts, Mat curr_frame, Mat prev_frame);
std::vector<int> clearLocalMins(int* minPositions, int* values, int n);
std::vector<Region> cutRegionsUpDown(std::vector<Region> regions,cv::Mat frame);
void splitIntoImages(Mat image,Mat binImage,std::vector<int> new_local_mins,std::vector<cv::Mat> &charImages,std::vector<cv::Mat> &charBinImages);



// Функции алгоритма
std::vector<Region> processFrameRegions(cv::Mat frame,std::vector<Region> regions,std::vector<cv::Mat> &united_images);
cv::Mat searchForNumber(cv::Mat image,cv::Mat prev_frame,std::vector<cv::KeyPoint> curr_keypts,
std::vector<cv::KeyPoint> prev_keypts);
void myMATLABLocalization(char* filename);




////////////////////////////////////////////  MAIN  ////////////////////////////////////////////  MAIN
////////////////////////////////////////////  MAIN  ////////////////////////////////////////////  MAIN

int _main()
{
	// getVideoTrack()
	// "C:\\Main\\WORK\\video\\new\\new25.avi"
	myMATLABLocalization(getVideoTrack()); 
	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

class Region 
{
public:
std::vector<cv::KeyPoint> keypts;
cv::RotatedRect box;
cv::Mat box_image;
cv::Rect rect;
bool isGood;

Region(std::vector<cv::KeyPoint> keypoints,cv::Mat frame)
{
keypts = keypoints;
std::vector<cv::Point> points;
for (int j = 0; j < keypoints.size(); j++)
{
points.push_back((cv::Point)keypoints.at(j).pt);
}

box = getMinBBox(frame,points);
box_image = getBBoxImage(frame,box); 

rect = boundingRect(Mat(points));
isGood = true;
}
Region(cv::Rect inp_rect)
{ // новый конструктор
rect = inp_rect; // ???????????????????????????????????????????????????????? be careful !!!
}
void drawRect(cv::Mat &image,cv::Scalar color = cv::Scalar(0,255,0),int thickness = 2)
{
if (rect.area() > 0)
{
rectangle(image,rect,color,thickness);
//char msg[50];
//std::sprintf(msg,"x=%d,y=%d,w=%d,h=%d;",rect.x,rect.y,rect.width,rect.height);
//cv::putText(image,msg,Point(rect.x,rect.y+rect.height/2),FONT_HERSHEY_PLAIN, .8, cv::Scalar(255,0,0));
}
}
cv::Mat getRectImage(cv::Mat image)
{
cv::Mat imROI = image(rect);
return imROI;
}
cv::Mat getKeypointsMap()
{
cv::Mat map(cv::Size(rect.width,rect.height),CV_8U,cv::Scalar(0));
for (int i = 0; i < keypts.size(); i++)
{
cv::Point pt = keypts.at(i).pt;
pt.x -= rect.x; pt.y -= rect.y;
cv::circle(map,pt,2,cv::Scalar(255),2);
}
return map;
}
float getKeyptsDensity()
{ // плотность кейпойнтов в прямоуг-ке.
return ( rect.area() / keypts.size() );
}
float getMorphoDensity(cv::Mat frame)
{
cv::Mat tmp = frame(rect);
cv::Mat tmp2;
tmp.copyTo(tmp2);
return calcMatMorphoDensity(tmp2);
}
float getBiggerMorphoDensity(cv::Mat frame)
{
// Померять плотность морф.градиента в увеличенном во все стороны на 25% прямоугольнике.
Point p1(rect.x,rect.y),
 p2(rect.x+rect.width,rect.y+rect.height);
if (p1.x > rect.width*0.25) p1.x-=rect.width*0.25; else p1.x = 0;
if (p1.y > rect.height*0.25) p1.y-=rect.height*0.25; else p1.y = 0;
if (p2.x < frame.cols - rect.width*0.25) p2.x+=rect.width*0.25; else p2.x = frame.cols-1;
if (p1.x < frame.rows - rect.height*0.25) p2.y-=rect.height*0.25; else p2.y = frame.rows-1;
Rect big_rect(p1,p2);
cv::Mat tmp = frame(big_rect);
cv::Mat tmp2;
tmp.copyTo(tmp2);
return calcMatMorphoDensity(tmp2);
}
float getDiffUpDownMorphoDensity(cv::Mat frame)
{
// модуль разницы между плотностями морф. град-та 
// в верхней и нижней половинах изображения
cv::Mat tmp2;
frame.copyTo(tmp2);

printf("\nframe.rows=%d, cols=%d",frame.rows,frame.cols);
Rect up_rect(Point(rect.x,rect.y),Point(rect.x+rect.width,rect.y+rect.height/2));
Rect down_rect(Point(rect.x,rect.y+rect.height/2),Point(rect.x+rect.width,rect.y+rect.height));
printf("\nup.x=%d,up.y=%d,up.width=%d,up.height=%d",up_rect.x,up_rect.y,up_rect.width,up_rect.height);

printf("\ndown.x=%d,down.y=%d,down.width=%d,down.height=%d",down_rect.x,down_rect.y,down_rect.width,down_rect.height);
float dens1 = calcMatMorphoDensity(tmp2(up_rect));
float dens2 = calcMatMorphoDensity(tmp2(down_rect));
return abs(dens1 - dens2);
//return 0.;
}
};

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

// CURRENT DEVELOPMENT // CURRENT DEVELOPMENT // CURRENT DEVELOPMENT // CURRENT DEVELOPMENT
// CURRENT DEVELOPMENT // CURRENT DEVELOPMENT // CURRENT DEVELOPMENT // CURRENT DEVELOPMENT

bool checkIfText(Region region,cv::Mat frame)
{ // TO DO - написать что-то поадекватнее!
Mat img = frame(region.rect);
Mat copy;
img.copyTo(copy);
prewittFilter(copy,copy);
if (whiteRatio(copy) <= 20.0)
return false;

return true;
}


//std::vector<Mat> joinImages(std::vector<Mat> charImages,std::vector<bool> joinMap)
//{ // будет объединять charBinImage's в порядке, указанном картой. 
//	std::vector<Mat> new_charImages;
//	
//	for (int i = 0; i < charImages.size()-1; i++)
//	{ // есть ли объединение i-го и следующего
//	if (joinMap.at(i))
//	{ 
//	new_charImages.push_back(concatImages(charImages.at(i),charImages.at(i+1)));
//	i++;
//	}
//	else 
//	{	
//	new_charImages.push_back(charImages.at(i));	
//	}
//	}
//
//	if ( !joinMap.at(joinMap.size()-1) )
//	{	
//	new_charImages.push_back(charImages.at(charImages.size()-1));
//	}
//
//	return new_charImages;
//}


Mat drawNumberSeparated(std::vector<cv::Mat> charImages,std::vector<cv::Mat> unitedImages,std::vector<int> unitedBegins,int curr_number_height)
{ 


std::vector<int> imgs_y,united_y; // координаты столбца, в который нужно вставить картинку (левый отступ от края)
int curr_otst1 = 5,curr_otst2 = 5; // будут хранить требуемую ширину для умещения всех charImages и unitedImages на одной картинке соответственно
for(int i = 0; i < charImages.size(); i++)
{
imgs_y.push_back(curr_otst1);
curr_otst1 += charImages.at(i).cols + 25;
}
for (int i = 0; i < unitedImages.size(); i++)
{
united_y.push_back(imgs_y.at(unitedBegins.at(i))); // такое же начало, как и левого из родителей
}
if (unitedImages.size()>0){
curr_otst2 = united_y.at(united_y.size()-1) // начало последнего
+unitedImages.at(unitedImages.size()-1).cols+25; // ширина последнего + 25
}


// Создаем общее изображение для charImages в 1-й строчке и unitedImages - во второй
int rows = 2 * curr_number_height + 50;
int cols = curr_otst1 > curr_otst2 ? curr_otst1 : curr_otst2; 
cv::Mat bigImg(rows,cols,charImages.at(0).type());

// Отрисовываем все charImages и unitedImages
for(int i = 0; i < charImages.size(); i++)
{
Mat ROI = bigImg(cv::Rect(imgs_y.at(i),10,charImages.at(i).cols,charImages.at(i).rows));
(charImages.at(i)).copyTo(ROI);
}
for (int i = 0; i < unitedImages.size(); i++)
{
Mat ROI = bigImg(cv::Rect(united_y.at(i),20+curr_number_height+10,unitedImages.at(i).cols,unitedImages.at(i).rows));	
(unitedImages.at(i)).copyTo(ROI);
}

return bigImg;
}


bool canBeJoined(Mat bin1, Mat bin2)
{
if (whiteRatio(bin1) * whiteRatio(bin2) == 0) 
return false;
Mat joined = concatImages(bin1,bin2);

if (getBinImgHeight(joined) < 1.3 * getBinImgWidth(joined)) return false;
return true;
}

void processCharImages(std::vector<cv::Mat> &charImages, // вернутся новые - инвертированные и обрезанные
std::vector<cv::Mat> charBinImages,
std::vector<int> values,
Mat binImg,
std::vector<int> local_mins,
std::vector<cv::Mat> &unitedImages, // Объединяет те charImages, объед. которых удовл. ф-и canBeJoined,
std::vector<int> &unitedBegins // хранит порядк.номер левого из charImage'й, из которых он сделан
)
{ // решаем, что делать с кусками номера между лок.миним. 
std::vector<int> cbj; // cbj = canBeJoined, можно ли объединить i-й и (i+1)-й 
for (int i = 0; i < charBinImages.size()-1; i++)
{
if (canBeJoined(charBinImages.at(i),charBinImages.at(i+1)))
cbj.push_back(1);
else
cbj.push_back(0);
}
std::vector<cv::Mat> unitedBinImages;
int curr_number_height = charImages.at(0).rows;

for (int i = 0; i < local_mins.size(); i++)
{
if (cbj.at(i)==1)
{
Mat united = concatImages(charImages.at(i),charImages.at(i+1));
Mat unitedBin = concatImages(charBinImages.at(i),charBinImages.at(i+1));
unitedImages.push_back(united);
unitedBegins.push_back(i);
unitedBinImages.push_back(unitedBin);
}
}

// Обрезка charImages и unitedImages по черному с трех сторон 
std::vector<cv::Mat> cut_charImages;
std::vector<cv::Mat> cut_unitedImages;
for (int i = 0; i < charImages.size(); i++)
{
cut_charImages.push_back(cutCharWhite(charImages.at(i),charBinImages.at(i)));
}
for (int i = 0; i < unitedImages.size(); i++)
{
cut_unitedImages.push_back(cutCharWhite(unitedImages.at(i),unitedBinImages.at(i)));
}
charImages.swap(cut_charImages);
unitedImages.swap(cut_unitedImages);
// Инвертирование
for (int i = 0; i < charImages.size(); i++)
{
charImages.at(i) = invertImage(charImages.at(i));
}
for (int i = 0; i < unitedImages.size(); i++)
{
unitedImages.at(i) = invertImage(unitedImages.at(i));
}

Mat tmp = drawNumberSeparated(charImages,unitedImages,unitedBegins,curr_number_height);
tmpShow(tmp);
}


std::vector<int> recognize(cv::Mat number_image)
{
Mat image;
number_image.copyTo(image);
Mat binImage;
image = binarize(image);
image.copyTo(binImage);
number_image.copyTo(image);


int* histArr = getVertHist(binImage); // вычислить вертикальную гистограмму
minPositions = localMins(histArr,binImage.cols); // вычислить локальные минимумы на ней

int* values = new int[NUM_MINS];
for (int i = 0; i < NUM_MINS; i++)
{
values[i] = histArr[minPositions[i]];
}

std::vector<int> new_local_mins = clearLocalMins(minPositions, values, NUM_MINS);
std::vector<int> new_values;
// создать вектор хранящий значения в новых.лок.мин-х
for (int i = 0; i < new_local_mins.size(); i++) 
{
new_values.push_back(histArr[new_local_mins.at(i)]);
}


std::vector<cv::Mat> charImages; // куски изображения между двумя минимумами, предположительно - символ
std::vector<cv::Mat> charBinImages; // то же ЧБ
// разбить изображение номера и бин.номера на отдельные картинки
splitIntoImages(image,binImage,new_local_mins,charImages,charBinImages);

// Проверка на то, есть ли хоть один локальный минимум == 0. Если да, распознаем, нет - адьос.
int mult = 1;
for (int i = 0; i < new_values.size(); i++)
mult *= new_values.at(i);
std::vector<Mat> unitedImages;
std::vector<int> unitedBegins;
if (mult == 0)
processCharImages(charImages,
charBinImages,
new_values,
binImage,
new_local_mins,
unitedImages,
unitedBegins);

// TO DO - подготовка к распознаванию, распознавание, обработка результатов!


//drawLocalMins(binImage,new_local_mins);
//tmpShow(binImage);
//tmpShow(image);

std::vector<int> recognized_number;
return recognized_number;
}




// CURRENT DEVELOPMENT  END // CURRENT DEVELOPMENT  END // CURRENT DEVELOPMENT  END // CURRENT DEVELOPMENT  END 
// CURRENT DEVELOPMENT  END // CURRENT DEVELOPMENT  END  // CURRENT DEVELOPMENT END  // CURRENT DEVELOPMENT END 


// Функции алгоритма
void myMATLABLocalization(char* filename)
{
// BEGIN Prepare for playing video 
capture.open(filename); // from file


int num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
if (num_frames > 0)
VIDEO_TRACKBAR_MAX_VALUE=num_frames;
else
VIDEO_TRACKBAR_MAX_VALUE = 1000;

double rate = capture.get(CV_CAP_PROP_FPS); // frames per second
bool stop(false);
cv::Mat frame,prev_frame,background_frame;

std::vector<cv::KeyPoint> curr_keypts,prev_keypts,
background_keypts; // кейпойнты фона (нужно будет делать снимок фона перед проездом поезда и вычислять кейпойнты)

// Declaring windows
cv::namedWindow("Extracted Frame 1");
cv::namedWindow("Settings");

int frameNumber = 0;
int delay = 1000/rate; // 1000 microseconds = 1 sec => delay is real time needed for one frame in current avi.
// END
capture.read(frame);
widenFrame(frame,15);
//// Setup output video
//cv::VideoWriter output_cap("C:\\Main\\WORK\\!test\\current_wagons.avi", 
//	  capture.get(CV_CAP_PROP_FOURCC),
//	  capture.get(CV_CAP_PROP_FPS),
//	  cv::Size(frame.cols,frame.rows));
//// int codec = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
//if (!output_cap.isOpened())
//{
//	std::cout << "!!! Output video could not be opened" << std::endl;
//	return;
//}

// Параметры, зависящие от конкретного видео
DOWN_BORDER = frame.rows;
MIN_BBOX_AREA = NUMBER_HEIGHT * NUMBER_HEIGHT; 

cv::moveWindow("Extracted Frame 1",200,50);
cv::moveWindow("Extracted Frame 2",200,130+frame.rows);
cv::moveWindow("Histogram",200+frame.cols+20,130+frame.rows);
cv::moveWindow("Settings",200+frame.cols+20,50);

// Adding trackbars...
createTrackbar("Frame No", "Settings", &VIDEO_TRACKBAR_VALUE, VIDEO_TRACKBAR_MAX_VALUE, on_trackbar_listener);
createTrackbar("FAST_THRES", "Settings", &FASTTHRESH_TRACKBAR_VALUE, FASTTHRESH_TRACKBAR_MAX_VALUE, on_trackbar2_listener);
createTrackbar("RADIUS_X", "Settings", &RADIUS_X_TRACKBAR_VALUE, RADIUS_TRACKBAR_MAX_VALUE, on_trackbar3_listener);
createTrackbar("RADIUS_Y", "Settings", &RADIUS_Y_TRACKBAR_VALUE, RADIUS_TRACKBAR_MAX_VALUE, on_trackbar3_listener);
createTrackbar("COMPR_RATE", "Settings", &COMPR_RATE_TRACKBAR_VALUE, COMPR_RATE_TRACKBAR_MAX_VALUE, on_trackbar3_listener);
setMouseCallback( "Extracted Frame 1", onMouse, 0 );

printf("NUMBER_HEIGHT=%d;\n",NUMBER_HEIGHT);

num_frame = 0;

while (!stop) 
{
frame.copyTo(prev_frame);
std::swap(curr_keypts, prev_keypts);

num_frame++;

cv::setTrackbarPos("Frame No","Settings",1+getTrackbarPos("Frame No","Settings"));

if (!capture.read(frame))
break;

widenFrame(frame,15);
cv::Mat image1(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0));
cv::Mat image2(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0));
frame.copyTo(image1);
frame.copyTo(image2);
curr_keypts = detectCorners(frame,COMPR_RATE,CORNERS_THRESH);
if (num_frame == 1) // считаем, что на первом кадре ФОН, тогда навсегда сохраним изображение фона и его кейпойнты
{
frame.copyTo(background_frame);
background_keypts=detectCorners(frame,COMPR_RATE,5); 
}

if (num_frame > 2 )//&& prev_keypts.size() > 0) 
{	
curr_keypts = cleanStaticKeypts(curr_keypts,background_keypts,frame,background_frame); // почистить от кейпойнтов фона
if ( prev_keypts.size() > 0 )
{
//curr_keypts = cleanStaticKeypts(curr_keypts,prev_keypts,frame,prev_frame); // почистить от кейпойнтов предыдущего кадра
}
}

image2 = searchForNumber(image2,prev_frame,curr_keypts,prev_keypts);
int key = cv::waitKey(delay/5); 
if (key == 27) // ESC pressed - close window
{
stop = true;
}
else if (key == 32) // SPACE pressed - pause until any key pressed
{	
cv::waitKey(0);
}
else if (key == 9)
{
cv::putText(image2,"SETTING UP BORDERS",Point(10,image1.rows/2), FONT_HERSHEY_SIMPLEX, .9, CV_RGB(0,0,0));
cv::imshow("Extracted Frame 1",image2);
borders_setting = 1;
cv::waitKey(0);
}
drawBorders(image2);
//cv::imshow("Extracted Frame 1",image1);
cv::imshow("Extracted Frame 1",image2);

//output_cap.write(image2);
}
capture.release();
//output_cap.release();
}

cv::Mat searchForNumber(cv::Mat image,cv::Mat prev_frame,std::vector<cv::KeyPoint> curr_keypts,
std::vector<cv::KeyPoint> prev_keypts) 
{ 
printf("\n\nFrame No:%d",num_frame);
double	t;	// time
std::vector<Region> regions; // хранит все области-кандидаты данного кадра
cv::Mat cleanImage;
image.copyTo(cleanImage);

std::vector<cv::KeyPoint> keypoints;
clearKeypointsByAreaHeight(curr_keypts,keypoints); // отсеять keypoint'ы по огранич. высоте 

// Отрисовка keypoints
t = (double)getTickCount();
cv::drawKeypoints(image,keypoints,image,cv::Scalar(0,255,0),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
t = ((double)getTickCount() - t) / getTickFrequency();
printf("\nKeypoints drawing time: %0.5f",t);

// Объединение keypoints в группы по связности
t = (double)getTickCount();
std::vector<std::vector<cv::KeyPoint>> kp_groups;
if (keypoints.size() > 0)
{
kp_groups = getKeypointGroups(keypoints,RADIUS_X,RADIUS_Y);
}
t = ((double)getTickCount() - t) / getTickFrequency();
printf("\nKeypoints processing time: %0.5f",t);


t = (double)getTickCount();
// Создание регионов по кейпойнтам (с ограничениями (*)) 
for (int i = 0; i < kp_groups.size(); i++)
{
if (kp_groups.at(i).size() >= 4) // (*) а) минимальное кол-во точек для создания региона = 4
{
Region tmp_region(kp_groups.at(i),cleanImage);
if(	tmp_region.box.size.area() > MIN_BBOX_AREA) //  (*) учитываем только регионы с площадью box'a не менее чем заданная
{
regions.push_back(tmp_region);
}
}
}

// Обработка регионов, отбрасывание лишних
std::vector<cv::Mat> united_images; // изображения, полученные объединением двух соседних по горизонтали регионов
regions = processFrameRegions(cleanImage,regions,united_images);
for (int i = 0; i < regions.size(); i++)
{
Region tmp_region = regions.at(i);
if ( tmp_region.rect.height > NUMBER_HEIGHT * h_alpha_down && 
tmp_region.rect.height < NUMBER_HEIGHT * h_alpha_up &&
tmp_region.rect.width > NUMBER_HEIGHT * 3.5 
&& checkIfText(tmp_region,cleanImage)) // удовлетворяет ли этот прямоугольник ограничениям
{
tmp_region.drawRect(image);
recognize(cleanImage(tmp_region.rect));
// + инвертированное
Mat inv = invertImage(cleanImage(tmp_region.rect));
recognize(inv);

char filename[50];
std::sprintf(filename,"C:\\Main\\WORK\\!test\\%d-%d.bmp",num_frame,i);
//cv::imwrite(filename,concatImages(recogn_im,inv_recogn_im));
}
else
{
tmp_region.drawRect(image,cv::Scalar(0,0,255),2);
}

//cv::Mat bin = binarize(cleanImage(tmp_region.rect));
//bin.copyTo(image(tmp_region.rect)); 

/*char filename[50];
std::sprintf(filename,"C:\\Main\\WORK\\!test\\%d-%d.bmp",num_frame,i);
cv::imwrite(filename,cleanImage(tmp_region.rect));*/
}

for (int i = 0; i < united_images.size(); i++)
{
if ( united_images.at(i).rows > NUMBER_HEIGHT * h_alpha_down && 
united_images.at(i).rows < NUMBER_HEIGHT * h_alpha_up &&
united_images.at(i).cols > NUMBER_HEIGHT * 3.5) // удовлетворяет ли этот прямоугольник ограничениям
{
if (true) // CHECK IF TEXT!!!
{
recognize(united_images.at(i));
// + инвертированное
Mat inv = invertImage(united_images.at(i));
recognize(inv);
}
}
}



t = ((double)getTickCount() - t) / getTickFrequency();
printf("\nAreas processing (incl. drawing) time: %0.5f",t);


return image;
} 

std::vector<Region> processFrameRegions(cv::Mat frame,std::vector<Region> regions,std::vector<cv::Mat> &united_images)
{
std::vector<Region> new_regions;

// а) Проверка на ПЕРЕСЕЧЕНИЕ  регионов 
// если пересекаются - объединить в один
new_regions = getJoinedRegions(regions,frame);


// б) Обрезает лишнее сверху и снизу
new_regions = cutRegionsUpDown(new_regions,frame);
for (int i = 0; i < new_regions.size(); i++)
{
new_regions.at(i).rect = widenRect(new_regions.at(i).rect,3);
}

// в) Разрезать на слова
// cutIntoWords !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


// г) Проверка на наличие двух соседних регионов
// позволяет соединять части номера в одно изображение
united_images = getUnitedImages(new_regions,frame);
/*for (int i = 0; i < united_images.size(); i++)
{
char filename[50];
std::sprintf(filename,"C:\\Main\\WORK\\!test\\united_%d-%d.bmp",num_frame,i);
cv::imwrite(filename,united_images.at(i));
}*/

return new_regions;
}

// Важные функции
std::vector<cv::KeyPoint> detectCorners(cv::Mat image/* входящее изображение*/, float ratio /* коэфф сжатия высоты и ширины*/,int fast_thresh /*порог для детекции уголков*/) 
{
// vector of keypoints
std::vector<cv::KeyPoint> keypoints;
// corners detector
cv::FastFeatureDetector fast(
fast_thresh); // threshold for detection

cv::Mat small_image;
cv::resize(image,small_image,cv::Size(image.cols/ratio,image.rows/ratio));
fast.detect(small_image,keypoints);
// drawing keypoints on the image
cv::Mat blackIm(image.rows,image.cols,CV_8UC3,cv::Scalar(0));
for (int j = 0; j < (int)keypoints.size(); j++)
{
// change keypoints coords * ratio
keypoints.at(j).pt.x *= ratio;
keypoints.at(j).pt.y *= ratio;
}
return keypoints;
}
std::vector<std::vector<cv::KeyPoint>> getKeypointGroups(std::vector<cv::KeyPoint> keypoints,int RADIUS_X,int RADIUS_Y)
{
std::vector<std::vector<cv::KeyPoint>> kp_groups;

const int N = (int)keypoints.size();
int i;
int* id = new int[N]; // будет хранить "связные" области (как в алгоритме из книги)
for (i = 0; i < N; i++) id[i] = i;
// Перебор вершин
for (int j = 0; j < N-1; j++)
for (int k = j+1; k < N; k++)
{
float X = keypoints.at(j).pt.x-keypoints.at(k).pt.x;
float Y = keypoints.at(j).pt.y-keypoints.at(k).pt.y;
//if ( (int)pow(X,2) + (int)pow(Y,2) < (int)pow((float)RADIUS,2)) // СТАРАЯ ВЕРСИЯ - ПОИСК В КРУГЕ. 
if ( (int)(pow(X,2) / pow((float)RADIUS_X,2)) + (int)(pow(Y,2) / pow((float)RADIUS_Y,2)) < 1 ) // эллипс
{
//printf("This keypoints are connected - %d and %d\n",j,k);
//printf("j=%d; k=%d; X=%f; Y=%f; sumsquared=%d; RADIUS^2=%d\n",j,k,X,Y,(int)pow(X,2) + (int)pow(Y,2),(int)pow((float)RADIUS,2));
int t = id[j];
if (t == id[k]) continue;
for (i = 0; i < N; i++)
if (id[i] == t) id[i] = id[k];
}
}

std::vector<int> groups; // будет хранить уникальные номера групп (Пр. 1 25 - то в id есть только все элементы равны или 1, или 25, т.е. образуют две связные области)
for (i = 0; i < N; i++)
{   
bool unique = true;
for (int j = 0; j < groups.size(); j++)
{
if (id[i] == groups.at(j))
unique = false;
}
if (unique)
groups.push_back(id[i]);
}

//// для каждой группы создать вектор keypoint-ов, состоящий только их keypoint-ов этой группы
for (int j = 0; j < groups.size(); j++)
{
std::vector<cv::KeyPoint> tmp;
// Проход по точкам
for (i = 0; i < N; i++)
{
if (id[i] == groups.at(j))
tmp.push_back(keypoints.at(i));
}
kp_groups.push_back(tmp);
}
return kp_groups;
}
void getTrueBoxWidthHeight(RotatedRect bBox,float &width,float &height)
{
// вовзращает настоящие высоту и ширину box-а
double angle = bBox.angle; // угол поворота
if (angle <= -45.)
angle += 90.;

if (angle < 0)
{
width = bBox.size.width;
height = bBox.size.height;
}
else if (angle > 0)
{
width = bBox.size.height;
height = bBox.size.width;
}
else
{
cv::Point2f vertices[4];
bBox.points(vertices);
if (vertices[0].x == vertices[1].x)
{
height = abs(vertices[0].y - vertices[1].y);
width = abs(vertices[0].x - vertices[2].x);
}
else
{
height = abs(vertices[0].y - vertices[2].y);
width = abs(vertices[0].x - vertices[1].x);
}
}
//printf("\nangle=%.2f,height=%.2f,width=%.2f",bBox.angle,height,width);
}
cv::RotatedRect getMinBBox(cv::Mat image,std::vector<cv::Point> points)
{
// найти минимальный описанный прямоугольник вокруг точек
cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));
return box;
}
cv::Mat drawMinBBox(cv::Mat image, cv::RotatedRect box,cv::Scalar color)
{
double angle = box.angle; // угол поворота
if (angle <= -45.)
angle += 90.;

cv::Point2f vertices[4];
box.points(vertices);
for(int i = 0; i < 4; ++i)
{
cv::line(image, vertices[i], vertices[(i + 1) % 4], color, 1, CV_AA);
}

float height,width;
getTrueBoxWidthHeight(box,width,height);

char msg[30];

std::sprintf(msg, "(%.1f,%.1f,%.1f)", angle,height,width);

putText(image, msg, (cv::Point)box.center, FONT_HERSHEY_SIMPLEX, .4, CV_RGB(255,0,0));
return image;
}
bool checkedMinBBox(cv::RotatedRect box)
{
// ПРОВЕРКА BOX-a:
// 1) угол наклона  
// 2) высота
// 3) положение в кадре (от UP_BORDER до DOWN_BORDER)

float height,width;
getTrueBoxWidthHeight(box,width,height);
double angle = box.angle; // угол поворота
if (angle <= -45.)
angle += 90.;

if (angle > ANGLE_THRESH || angle < -  ANGLE_THRESH) // отсеять по углу наклона
return false;

if (height > NUMBER_HEIGHT*h_alpha_up || height < NUMBER_HEIGHT * h_alpha_down)
return false;

if (box.center.y < UP_BORDER || box.center.y > DOWN_BORDER)
return false;
return true;
}
void clearKeypointsByAreaHeight(std::vector<cv::KeyPoint> keypoints,std::vector<cv::KeyPoint> &out_keypoints)
{
out_keypoints.clear();
for (int i = 0; i < keypoints.size(); i++)
{
if (keypoints.at(i).pt.y > UP_BORDER && keypoints.at(i).pt.y < DOWN_BORDER)
out_keypoints.push_back(keypoints.at(i));
}
}
cv::Mat getBBoxImage(cv::Mat image,cv::RotatedRect box)
{// ПОЛУЧЕНИЕ РЕАЛЬНОЙ КАРТИНКИ ИЗ ИЗОБРАЖЕНИЯ ПО bbox-у
// АЛГОРИТМ - 1) ПОЛУЧИТЬ ПОВЕРНУТОЕ ИЗОБРАЖЕНИЕ, 
// 2) ПОЛУЧИТЬ КООРДИНАТЫ ПРЯМОУГОЛЬНИКА НА ПОВЕРНУТОМ,
// 3) ВЫДЕЛИТЬ НА НЕМ ROI и вернуть
cv::Mat imgROI;
image.copyTo(imgROI);
double angle = box.angle; // угол поворота
if (angle <= -45.)
angle += 90.;
imrotate(imgROI,imgROI,angle);
cv::Mat rot_mat = getRotationMatrix2D(Point(imgROI.cols/2,imgROI.rows/2),angle,1);
cv::Point2f vertices[4],new_vertices[4];
box.points(vertices);
for (int i = 0; i < 4; i++)
{
new_vertices[i].x = abs(rot_mat.at<double>(0,0) * vertices[i].x + rot_mat.at<double>(0,1) * vertices[i].y + rot_mat.at<double>(0,2)); 
new_vertices[i].y = abs(rot_mat.at<double>(1,0) * vertices[i].x + rot_mat.at<double>(1,1) * vertices[i].y + rot_mat.at<double>(1,2)); 
}
cv::Rect rect(new_vertices[0],new_vertices[2]);
if (rect.x < 0 || rect.x+rect.width >= imgROI.cols-1 || rect.y < 0 || rect.y+rect.height >= imgROI.rows) 
{
// если область вылазит за пределы изображения, вернуть один пиксел
cv::Mat errIm(1,1,CV_8U);
return errIm;
}
else
{ // если удовлетворяет - выделяем ее на изображени
imgROI = imgROI(rect);	
}

//cv::Mat box_image = image(box);
return imgROI;
}
std::vector<Region> getJoinedRegions(std::vector<Region> regions,cv::Mat frame)
{
const int N = regions.size();
int* id = new int[N]; // будет хранить "связные" области (как в алгоритме из книги)
for (int i = 0; i < N; i++) id[i] = i;
for (int j = 0; j < N-1; j++)
for (int k = j+1; k < N; k++)
{
if ( checkRectOverlap(regions.at(k).rect,regions.at(j).rect) ) 
{
int t = id[j];
if (t == id[k]) continue;
for (int i = 0; i < N; i++)
if (id[i] == t) id[i] = id[k];
}
}

std::vector<Region> new_regions;
for (int j = 0; j < regions.size(); j++)
{
int group_no = id[j];
if (!regions.at(j).isGood) continue; // если еще не участвовал в объединении
std::vector<cv::KeyPoint> tmp_keypts = regions.at(j).keypts; // хранит все кейпойнты объедин-го региона
for (int k = j+1; k < regions.size(); k++)
{
if (id[k] == group_no && regions.at(k).isGood)
{
tmp_keypts.insert(tmp_keypts.end(),regions.at(k).keypts.begin(),regions.at(k).keypts.end());
regions.at(k).isGood = false; // уже участвовал в объединении, больше не нужен
}
}

if (tmp_keypts.size() == regions.at(j).keypts.size())
{
new_regions.push_back(regions.at(j));
}
else
{
Region reg(tmp_keypts,frame);
new_regions.push_back(reg);
}
}

return new_regions;
}
std::vector<cv::Mat> getUnitedImages(std::vector<Region> regions,cv::Mat frame)
{
std::vector<cv::Mat> united_images;
// ПРОВЕРКА, НЕТ ЛИ РЯДОМ СПРАВА ИЛИ СЛЕВА РЕГИОНА С СОПОСТАВИМОЙ ВЫСОТОЙ, ЕСЛИ ЕСТЬ СОЗДАТЬ ОБЪЕДИНЕННОЕ ИЗОБРАЖЕНИЕ
for (int i = 0; i < regions.size(); i++)
{
for (int j = i+1; j < regions.size(); j++)
{
Rect rect1 = regions.at(j).rect;
Rect rect2 = regions.at(i).rect;
// Условия на объединяемые прямоугольники
if (
(rect1.y + rect1.height/2 > rect2.y && rect1.y + rect1.height/2 < rect2.y + rect2.height) && //середина одного не выходит за верт. границы другого (чтобы друг напротив друга)

(abs(rect1.x+rect1.width - rect2.x) < NUMBER_HEIGHT*2 ||                      // небольшое расстояние по горизонтали
abs(rect2.x+rect2.width - rect1.x) < NUMBER_HEIGHT*2) &&

(rect1.width < NUMBER_HEIGHT * 3 || rect2.width < NUMBER_HEIGHT * 3) &&       // один из них более узкий, чем три высоты номера 
!checkRectOverlap(rect1,rect2) &&                                             // не пересекаются
(rect1.height > 0.8*NUMBER_HEIGHT && rect2.height > 0.8 * NUMBER_HEIGHT) &&   // высота не ниже чем
(rect1.height < 2 * NUMBER_HEIGHT && rect2.height < 2 * NUMBER_HEIGHT)    &&    // и не выше чем
( rect1.width > 2 * NUMBER_HEIGHT || rect2.width > 2 * NUMBER_HEIGHT  )  &&   // один из них шире, чем две высоты номера
 rect1.width < frame.cols * 0.66 &&  rect2.width < frame.cols * 0.66 // каждая область не шире чем 2/3 ширины кадра
) 
{
cv::Mat im1;
cv::Mat im2;
if (rect1.x < rect2.x)
{
im1 = frame(rect1);
im2 = frame(rect2);
}
else
{
im1 = frame(rect2);
im2 = frame(rect1);
}

cv::Mat tmp1,tmp2;
im1.copyTo(tmp1);
im2.copyTo(tmp2);
cv::Mat res = concatImages(tmp1,tmp2); 

united_images.push_back(res);
}

}
}

return united_images;
}
std::vector<cv::KeyPoint> cleanStaticKeypts(std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts, Mat curr_frame, Mat prev_frame)
{
cv::Mat	  prev_desc, curr_desc;
std::vector<cv::DMatch>	matches;
cv::BruteForceMatcher<cv::Hamming>	matcher;
cv::BriefDescriptorExtractor	 extractor;
extractor.compute(curr_frame, curr_keypts, curr_desc);
extractor.compute(prev_frame, prev_keypts, prev_desc);
std::vector<cv::KeyPoint> pois;

if (prev_keypts.size() == 0)
return curr_keypts;

try // matcher throws an error because there aren't sometimes any matches.
{
matcher.match( curr_desc, prev_desc, matches);
}
catch (...)
{
printf("error");
}


// Пороги - смещение не менее чем и не более чем
int DIST_THOLD = 100;
float V_THOLD = 0.1f;
for (int i = 0; i < matches.size(); i++) 
{

if (matches[i].distance < DIST_THOLD) 
{
float x_d = curr_keypts[matches[i].queryIdx].pt.x -
prev_keypts[matches[i].trainIdx].pt.x;
float y_d = curr_keypts[matches[i].queryIdx].pt.y -
prev_keypts[matches[i].trainIdx].pt.y;

if (sqrt(x_d * x_d + y_d * y_d) > V_THOLD) 
{
pois.push_back(curr_keypts[matches[i].queryIdx]);
}
}
}

return pois;
}
std::vector<int> clearLocalMins(int* minPositions, int* values, int n)
{ // если два лок.минимума в гистограмме номера ближе, чем MIN_DIST пикс., 
//то оставить тот из них, значение в котором меньше.
int MIN_DIST = 5;
std::vector<int> new_local_mins;
int i = 0;
while (i < n)
{
if (i == n - 1) // последнее записываем по умолч.
{
new_local_mins.push_back(minPositions[i]);
break;
}
if (minPositions[i+1] - minPositions[i] < MIN_DIST) // расст между текущим и след миним-ми меньше MIN_DIST
{ // проверить значение в каком из них меньше
if (values[i] <= values[i+1])
{ // добавить i-й
new_local_mins.push_back(minPositions[i]);
i++; // чтобы пропустить в будущем i+1-й элемент
}
else
{ // добавить i+1-й
new_local_mins.push_back(minPositions[i+1]);
i++;
}
}
else 
{ // добавить этот минимум в текущие
new_local_mins.push_back(minPositions[i]);
}
i++;
}

return new_local_mins;
}
std::vector<Region> cutRegionsUpDown(std::vector<Region> regions,cv::Mat frame)
{// Вернет новые регионы, разрезанные на вертикальные слова (или удалит, если неудачно)
std::vector<Region> new_regions;
std::vector<cv::Rect> new_rects;

for (int nReg = 0; nReg < regions.size(); nReg++) // для каждого старого региона
{ 
cv::Mat processedImage;
frame(regions.at(nReg).rect).copyTo(processedImage);
processedImage = binarize(processedImage);
cv::morphologyEx(processedImage,processedImage,MORPH_GRADIENT,cv::Mat(1,5,CV_8U));
cv::morphologyEx(processedImage,processedImage,MORPH_CLOSE,cv::Mat(1,processedImage.cols/3,CV_8U));
int n = processedImage.rows;
int* inp_arr = calcHist(processedImage);
inp_arr = medFiltHist(inp_arr,n);
minPositions = localMins(inp_arr,n,10);
int mean = 0;
for (int i = 0; i < n; i++)
{
mean += inp_arr[i];
}
mean = mean/n;

for (int i = 0; i < n; i++)
{
if (inp_arr[i] < 255*processedImage.cols / 5 )
//|| ( belongs(i,minPositions,NUM_MINS) && inp_arr[i] < mean / 2)) // по локальным минимумам
inp_arr[i] = 0;
}

// разбиение на горизонтальные области
std::vector<std::vector<int>> row_arrays;
std::vector<int> tmp_arr;
if (inp_arr[0] != 0) // если область началась сразу
tmp_arr.push_back(0);
for (int i = 1; i < n; i++)
{
if ((inp_arr[i] == 0 && inp_arr[i-1] != 0) // начались нули
|| (i==n-1 && inp_arr[i] != 0)) 
{
tmp_arr.push_back(i);
row_arrays.push_back(tmp_arr);
tmp_arr.clear();
}
else if ((inp_arr[i] != 0 && inp_arr[i-1] == 0)) // началась область
{
tmp_arr.push_back(i);
}
}
for (int i = 0; i < row_arrays.size(); i++)
{
if (row_arrays.at(i).size() == 2)
{
cv::Rect rect(regions.at(nReg).rect.x,
 regions.at(nReg).rect.y+row_arrays.at(i).at(0),
 regions.at(nReg).rect.width,
 row_arrays.at(i).at(1) - row_arrays.at(i).at(0));
Region tmp_reg(rect);
new_rects.push_back(rect);
new_regions.push_back(tmp_reg);
}
}//END разбиение на горизонтальные области
/*// Отрисовка обрезания
// Draw histogram image
cv::Mat histImg(n,2*n,CV_8U,cv::Scalar(0));

int maxValue = 255*processedImage.cols;

int intensity = 0;
for (int i = 0; i < n; i++)
{
intensity = (int)(2*n * inp_arr[i]/maxValue);
cv::line(histImg,cv::Point(0,i),
cv::Point(intensity,i),
cv::Scalar::all(255));
//printf("\nrow[%d]=%d;",i,intensity);
}
cv::Mat clearRegion;
frame(regions.at(nReg).rect).copyTo(clearRegion);
cv::cvtColor(clearRegion,clearRegion,CV_RGB2GRAY);
cv::Mat tmp = concatImages(concatImages(clearRegion,processedImage),histImg);

for (int i = 0; i < n; i++)
{
//printf("\ninp_arr[%d]=%d",i,inp_arr[i]);
if (inp_arr[i] == 0)
cv::line(tmp,cv::Point(0,i),
cv::Point(tmp.cols-1,i),
cv::Scalar::all(0));
}

// Нарисовать локальные минимумы на гистограмме.
//for (int i = 0; i < NUM_MINS; i++) //while (minPositions[i] > 0 && minPositions[i] < n)
//{
//	//if (inp_arr[minPositions[i]] < )
//	printf("\n%d",inp_arr[minPositions[i]]);
//	cv::line(tmp,cv::Point(0,minPositions[i]),
//	cv::Point(tmp.cols-1,minPositions[i]),
//	cv::Scalar::all(128));
//}

tmpShow(tmp);

cv::imwrite(filename,tmp);*/
}
return new_regions;
}
void splitIntoImages(Mat image,Mat binImage,std::vector<int> new_local_mins,std::vector<cv::Mat> &charImages,std::vector<cv::Mat> &charBinImages)
{// Сохранить по отдельности в картинки
charImages.clear(); charBinImages.clear();

Mat charImg,charBinImg;

image(Range(0,image.rows-1),Range(0,new_local_mins.at(0))).copyTo(charImg);
charImages.push_back(charImg);
binImage(Range(0,image.rows-1),Range(0,new_local_mins.at(0))).copyTo(charBinImg);
charBinImages.push_back(charBinImg);
charImg.release();
charBinImg.release();
for (int i = 0; i < new_local_mins.size()-1; i++)
{
Mat charImg,charBinImg;
image(Range(0,image.rows-1),Range(new_local_mins.at(i),new_local_mins.at(i+1))).copyTo(charImg);
charImages.push_back(charImg);
binImage(Range(0,image.rows-1),Range(new_local_mins.at(i),new_local_mins.at(i+1))).copyTo(charBinImg);
charBinImages.push_back(charBinImg);
}
image(Range(0,image.rows-1),
Range( new_local_mins.at(new_local_mins.size()-1) , image.cols-1)).copyTo(charImg); 
charImages.push_back(charImg);
binImage(Range(0,image.rows-1),
Range( new_local_mins.at(new_local_mins.size()-1) , image.cols-1)).copyTo(charBinImg); 
charBinImages.push_back(charBinImg);

}





// Временно неиспользуемые
std::vector<cv::RotatedRect> makeTracking(std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts,cv::Mat &curr_image,cv::Mat prev_image,
std::vector<cv::RotatedRect> prev_boxes)
{
std::vector<std::vector<cv::KeyPoint>> prev_box_groups,curr_box_groups; // будут хранить группы кейпойнтов для каждого minBBox'a
// Оставить из старых кейп-в только принадлежащие прямоугольникам, разбить их на группы
for (int ctr_box = 0; ctr_box < prev_boxes.size(); ctr_box++)
{
std::vector<cv::KeyPoint> tmp_group;
cv::Point2f vert[4];
prev_boxes.at(ctr_box).points(vert);
std::vector<cv::Point2f> vertices;
for (int i = 0; i < 4; i++) vertices.push_back(vert[i]);

for (int ctr_keypt = 0; ctr_keypt < prev_keypts.size(); ctr_keypt++)	
{
if (belongsToArea(prev_keypts.at(ctr_keypt).pt,vertices))
{
tmp_group.push_back(prev_keypts.at(ctr_keypt));
}
}
prev_box_groups.push_back(tmp_group);
}
std::vector<cv::DMatch> matches = getBriskMatches(curr_image,prev_image,curr_keypts,prev_keypts);
// нарисовать на старом старые прямоугольники - со временем удалить
for (int i = 0; i < prev_boxes.size(); i++) 
{
RotatedRect tmp_box = prev_boxes.at(i);
prev_image = drawMinBBox(prev_image,tmp_box,cv::Scalar(0,0,255));
}

cv::Mat img_matches; 
curr_image.copyTo(img_matches);
img_matches = drawBriskMatches(curr_image,prev_image,curr_keypts,prev_keypts,matches);
// Поиск соответствий точкам из prev_box_groups в curr_keypts и запись в curr_box_groups.
for (int ctr_gr = 0; ctr_gr < prev_box_groups.size(); ctr_gr++)
{
std::vector<cv::KeyPoint> tmp = prev_box_groups.at(ctr_gr);
std::vector<cv::KeyPoint> tmp2;
// Проход по каждой группе, поиск соответствий
for (int ctr = 0; ctr < tmp.size(); ctr++)
{
for (int i = 0; i < matches.size(); i++)
{
if (prev_keypts.at(matches.at(i).trainIdx).pt.x == tmp.at(ctr).pt.x && 
prev_keypts.at(matches.at(i).trainIdx).pt.y == tmp.at(ctr).pt.y
&& abs(tmp.at(ctr).pt.y - curr_keypts.at(i).pt.y) < HEIGHT_THRESH
&& abs(tmp.at(ctr).pt.x - curr_keypts.at(i).pt.x) < WIDTH_THRESH)
{
tmp2.push_back(curr_keypts.at(i));
}
}
}
curr_box_groups.push_back(tmp2);
}
printf("new boxes = %d; old boxes = %d; \n",curr_box_groups.size(), prev_box_groups.size());


std::vector<cv::RotatedRect> curr_boxes;
// вокруг каждой из новых групп сделать minBBox, нарисовать
for (int i = 0; i < curr_box_groups.size(); i++)
{
// Перевести в вектор точек points
std::vector<cv::Point> points;
for (int j = 0; j < curr_box_groups.at(i).size(); j++)
{
points.push_back(curr_box_groups.at(i).at(j).pt);
}
// нарисовать
printf("Points in group[%d]=%d\n",i,points.size());
if (points.size()>2)
{
cv::RotatedRect box = getMinBBox(img_matches,points);
img_matches = drawMinBBox(img_matches,box,cv::Scalar(255,255,0));
curr_boxes.push_back(box);
}
}

cv::namedWindow("win");
cv::imshow("win",img_matches);
return curr_boxes; // должно возвращать curr_boxes, но их пока нет.
}
std::vector<cv::DMatch> getBriskMatches(cv::Mat curr_frame, cv::Mat prev_frame, std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts)
{
cv::Mat	  prev_desc, curr_desc;
std::vector<cv::DMatch>	matches;
cv::BruteForceMatcher<cv::Hamming>	matcher;
cv::BriefDescriptorExtractor	 extractor;
extractor.compute(curr_frame, curr_keypts, curr_desc);
extractor.compute(prev_frame, prev_keypts, prev_desc);
try // matcher throws an error because there aren't sometimes any matches.
{
matcher.match( curr_desc, prev_desc, matches);
// инфо по текущему и предыдущему кадрам, ключесвым точкам и дескрипторам
//printf("\ncurr_desc.rows=%d,prev_desc.rows=%d,matches.size()=%d; curr_keyp.size()=%d; prev_keyp.size()=%d\n",curr_desc.rows,prev_desc.rows,matches.size(),curr_keypts.size(),prev_keypts.size());
//for (int i = 0; i < matches.size(); i++)
//{
//	printf("Matches[%d].trainIdx=%d\n",i,matches.at(i).trainIdx); // какая точка какой соответствует
//}
}
catch (...)
{
printf("error");
}
return matches;
}
cv::Mat drawBriskMatches(cv::Mat curr_frame, cv::Mat prev_frame, std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts,std::vector<cv::DMatch> matches)
{
cv::Mat img_match;
cv::drawMatches(curr_frame, curr_keypts, prev_frame, prev_keypts, matches, img_match);

// Пронумеровать точки текущего кадра
for (int i = 0; i < curr_keypts.size(); i++)
{
char msg[30];
std::sprintf(msg, "%d", i);
putText(img_match, msg, curr_keypts.at(i).pt, FONT_HERSHEY_SIMPLEX, .4, CV_RGB(255,0,0));
}

// Пронумеровать точки предыдущего кадра
for (int i = 0; i < prev_keypts.size(); i++)
{
cv::Point2f pt = prev_keypts.at(i).pt;
pt.x += curr_frame.cols;
char msg[30];
std::sprintf(msg, "%d", i);
putText(img_match, msg, pt, FONT_HERSHEY_SIMPLEX, .4, CV_RGB(255,0,255));
}

return img_match;
}
std::vector<cv::DMatch> cleanMatches(std::vector<cv::DMatch> matches,std::vector<cv::KeyPoint> curr_keypts,std::vector<cv::KeyPoint> prev_keypts,int thresh)
{
std::vector<cv::DMatch> tmp_matches;
for (int i = 0; i < matches.size(); i++)
{
if (abs(curr_keypts.at(i).pt.y - prev_keypts.at(matches.at(i).trainIdx).pt.y) < thresh ) // отклонение по вертикали не более чем на thresh.
{
tmp_matches.push_back(matches.at(i));
}
}
return tmp_matches;
}
cv::Mat drawLinesHoughDetection(cv::Mat image)
{
//cv::Mat image = cv::imread("C:\\Main\\WORK\\Laganiere Images\\road.jpg");
cv::Mat contours;
cv::Canny(image,contours,125,350);
// Hough transform for line detection
std::vector<cv::Vec2f> lines;
//cv::cvtColor(image,image,CV_RGB2GRAY);
cv::HoughLines(contours,lines,
1,3.14/180, // step size
80); // minimum number of votes

std::vector<cv::Vec2f>::const_iterator it = lines.begin();
while (it != lines.end())
{
float rho = (*it)[0]; // first element is distanse rho
float theta = (*it)[1]; // second element is angle theta

if (theta<3.14/4. || theta > 3. * 3.14 / 4.) { // vertical line
// point of intersection of the line with first row
cv::Point pt1(rho/cos(theta),0);
// point of intersection of the line with last row
cv::Point pt2((rho - image.rows * sin(theta))/cos(theta),image.rows);

//draw a white line
cv::line(image,pt1,pt2,cv::Scalar(255),1);
}
else // horisontal line
{
// point of intersection of the 
// line with the first column
cv::Point pt1(0,rho/sin(theta));
// point of intersection of the line with last column
cv::Point pt2(image.cols,(rho-image.cols*cos(theta))/sin(theta));
cv::line(image,pt1,pt2,cv::Scalar(255),1);
}
++it;
}
/*cv::namedWindow("win");
cv::imshow("win",image);
cv::waitKey(0);*/
return image;
}
cv::Mat getContours(cv::Mat image)
{
image = binarize(image);
std::vector<std::vector<cv::Point>> contours;
cv::findContours(image,
contours, // a vector of contours
CV_RETR_EXTERNAL, // retrieve the external contours
CV_CHAIN_APPROX_NONE); // all pixels of each contours
// Draw black contours on a white image
cv::Mat result(image.size(),CV_8U,cv::Scalar(255));
cv::drawContours(result,contours,
-1, // ALL CONTOURS
cv::Scalar(0), // in black
2); // thickness
return result;
}

// Вспомогательные функции
int* copyPartOfArray(int* inp_arr,int start,int finish)
{
// Вернет новый массив, содержащий в себе элементы массива inp_arr 
// с элемента start (включительно) по элемент finish (невключительно).
const int win_size = finish-start;
int* out_arr = (int *)malloc(win_size * sizeof(int));
for (int i = start; i < finish; i++)
{
out_arr[i-start] = inp_arr[i];
}
return out_arr;
}
int* localMins(int* arr,int n,int win_width)
{
int* localmins = (int *)malloc(n * sizeof(int));
NUM_MINS = 0; // количество минимумов на текущий момент
 
int* current_window; // текущий подмассив, в котором ищется локальный минимум.
for (int i = 0; i < n - win_width; i++)
{
current_window = copyPartOfArray(arr,i,i+win_width);
int* mins = min(current_window,win_width);
if (mins[1] != 0 && mins[1] != win_width-1 && localmins[NUM_MINS-1] != mins[1] + i && !belongs(mins[1] + i,localmins,NUM_MINS-1)) // локальный минимум в окне
{
if (!(arr[mins[1] + i] == 0 && arr[mins[1] + i - 1] == 0 && arr[mins[1] + i + 1] == 0))
{
localmins[NUM_MINS] = mins[1] + i;
NUM_MINS += 1;
}
}
}
for (int i = 1; i < n - 1; i++)
{
if (((arr[i] == 0 && arr[i-1] > 0)||(arr[i] == 0 && arr[i+1] > 0))&&(!belongs(i,localmins,NUM_MINS-1)))  // также к локальные минимумы добавляются "подножия" ([... 0 0 0 20 25 40 ...] и в другую сторону)
{	
localmins[NUM_MINS] = i;
NUM_MINS += 1;	
}
}
localmins = mysort(localmins,NUM_MINS);
return copyPartOfArray(localmins,0,NUM_MINS);
}
cv::Mat drawHist(cv::Mat image)
{	
int n = image.rows;
int* inp_arr = calcHist(image);

inp_arr = medFiltHist(inp_arr,n);
inp_arr = medFiltHist(inp_arr,n);
inp_arr = cleanHist(inp_arr,n,120); // все, что меньше 20-ти в гистограмме сделать 0 (из-за сглаживания завтыки)
minPositions = localMins(inp_arr,n,NUMBER_HEIGHT);

// Draw histogram image
cv::Mat histImg(n,n,CV_8U,cv::Scalar(255));
int maxValue = -1;
/*for (int i = 0; i < n; i++)
{
if (inp_arr[i] > maxValue)
maxValue = inp_arr[i];
}*/ // если хотим, чтобы масштабировалось по максимальному элементу гистограммы
maxValue = 255*image.cols;   // без масштабирования - максимум = 255 (белый пиксель) * количество пикселей в колонке

int intensity = 0;
for (int i = 0; i < n; i++)
{
intensity = (int)(n * inp_arr[i]/maxValue);
cv::line(histImg,cv::Point(0,i),
cv::Point(intensity,i),
cv::Scalar::all(0));
}

// Нарисовать локальные минимумы на гистограмме.
for (int i = 0; i < NUM_MINS; i++) //while (minPositions[i] > 0 && minPositions[i] < n)
{
cv::line(histImg,cv::Point(0,minPositions[i]),
cv::Point(n,minPositions[i]),
cv::Scalar::all(128));
//i++;
}
// ПОРОГ!!!
/*thresh = (int)(1.15 * mean(inp_arr,n));
printf("\nthresh=%d;\n",thresh);*/


// Нарисовать линию предела 1.15 на гистограмме
cv::line(histImg,cv::Point((int)(thresh*n/maxValue),0),cv::Point((int)(thresh*n/maxValue),image.rows),cv::Scalar::all(128));

histogram = inp_arr;

return histImg;
}
void imrotate(cv::Mat image,cv::Mat &out_image,double angle)
{
// Вокруг какой точки вращать
Point center = Point( image.cols/2, image.rows/2 );
double scale = 1; // уменьшения/увеличение
// Матрица поворота
cv::Mat rot_mat = getRotationMatrix2D( center, angle, scale );

/// Повернуть
warpAffine( image, out_image, rot_mat, image.size() );
}
void drawBorders(cv::Mat &image)
{ // Отрисовывает веритикальные границы области поиска, если они были заданы.
if (UP_BORDER > 0 || DOWN_BORDER < image.rows)
{
line(image,Point(0,UP_BORDER),Point(image.cols-1,UP_BORDER),Scalar(255,0,255),2);
line(image,Point(0,DOWN_BORDER),Point(image.cols-1,DOWN_BORDER),Scalar(255,0,255),2);
}
}
void tmpShow(cv::Mat image)
{
// Отображение во временном окне входящего изображения
cv::namedWindow("tmp");
cv::imshow("tmp",image);
cv::waitKey(0);
cv::destroyWindow("tmp");
}
float calcMatMorphoDensity(cv::Mat img)
{
cv::cvtColor(img,img,CV_RGB2GRAY);
equalizeHist(img,img); // аналог imadjust
cv::morphologyEx(img,img,MORPH_GRADIENT,cv::Mat(5,5,CV_8U));
int sumBrigth = 0;
for (int j = 0; j < img.cols; j++)
for (int i = 0; i < img.rows; i++)
{
sumBrigth = sumBrigth + (int)(img.at<uchar>(i,j));
}
return (float)sumBrigth / (img.rows * img.cols);
}
bool checkRectOverlap(cv::Rect rect1,cv::Rect rect2)
{
if (rect1.x > rect2.x + rect2.width ||
rect1.x + rect1.width < rect2.x ||
rect1.y > rect2.y + rect2.height ||
rect1.y + rect1.height < rect2.y
)
return false;
return true;
}
Region joinRegions(Region region1,Region region2,cv::Mat frame)
{
std::vector<cv::KeyPoint> united_keypts;
united_keypts.insert(united_keypts.end(), region1.keypts.begin(), region1.keypts.end());
united_keypts.insert(united_keypts.end(), region2.keypts.begin(), region2.keypts.end());
Region united_region(united_keypts,frame);
return united_region;
}
Rect widenRect(Rect rect,int pixels)
{// расширить Rect на Pixel пикселов во все стороны
Rect new_rect(rect.x-pixels,rect.y-pixels,rect.width+pixels*2,rect.height+pixels*2);
return new_rect;
}
cv::Mat concatImages(cv::Mat img1,cv::Mat img2) 
{ // Соединить два изображения в одно!
if (img1.rows > img2.rows) // выравнять высоту (растянуть меньшее до нужной высоты)
{
cv::resize(img2,img2,cv::Size(img2.cols,img1.rows));
}
else if (img1.rows < img2.rows)
{
cv::resize(img1,img1,cv::Size(img1.cols,img2.rows));
}

cv::Mat new_image(img1.rows,img1.cols+img2.cols,img1.type());
if (img1.channels() != img2.channels())
{
if (img1.channels() == 3)
cv::cvtColor(img1,img1,CV_RGB2GRAY);
if (img2.channels() == 3)
cv::cvtColor(img2,img2,CV_RGB2GRAY);
}

img1.copyTo(new_image.colRange(0,img1.cols));
img2.copyTo(new_image.colRange(img1.cols,img1.cols+img2.cols));

return new_image;
}
int getBinImgHeight(Mat charImage) 
{ // вычисляет высоту белой области в ч.б. картинке, "обрезая" снизу и сверху 
int row_sum, // сумма яркостей в строке
up_otst = -1,down_otst = -1; // высота черной области сверху и снизу

// вычисление вернего отступа
for (int i = 0; i < charImage.rows; i++)
{
row_sum = 0;
for (int j = 0; j < charImage.cols * charImage.channels(); j++)
{
row_sum += charImage.at<uchar>(i,j);
}

if (row_sum != 0)
{
up_otst = i;
break;
}
}

if (up_otst == -1) // если до конца так и не появилось НЕчерной строчки...
return 0;

// вычисление нижнего отступа
for (int i = 0; i < charImage.rows; i++)
{
row_sum = 0;
for (int j = 0; j < charImage.cols * charImage.channels(); j++)
{
row_sum += charImage.at<uchar>(charImage.rows-i-1,j);
}

if (row_sum != 0)
{
down_otst = charImage.rows - i - 1;
break;
}
}

return down_otst - up_otst + 1;
}
int getBinImgWidth(Mat charImage)
{
int col_sum, // сумма яркостей в столбце
left_otst = -1,right_otst = -1; // высота черной области слева и справа

for (int i = 0; i < charImage.cols * charImage.channels(); i++)
{
col_sum = 0;
for (int j = 0; j < charImage.rows; j++)
{
col_sum += charImage.at<uchar>(j,i);
}

if (col_sum != 0)
{
left_otst = i;
break;
}
}


if (left_otst == -1) 
return 0;

for (int i = 0; i < charImage.cols * charImage.channels() ; i++)
{
col_sum = 0;
for (int j = 0; j < charImage.rows; j++)
{
col_sum += charImage.at<uchar>(j,charImage.cols*charImage.channels()-i-1);
}


if (col_sum != 0)
{
right_otst = charImage.cols * charImage.channels() - i - 1;
break;
}
}


return (right_otst - left_otst + 1)/3;
}
void printIntVector(std::vector<int> vect)
{
printf("\n");
for (int i = 0; i < vect.size(); i++)
{
printf("%d ",vect.at(i));
}
printf("\n");
}
void drawLocalMins(Mat &image,std::vector<int> local_mins,cv::Scalar color)
{
// Нарисовать локальные минимумы на изображении номера
for (int i = 0; i < local_mins.size(); i++)
{
cv::line(image,cv::Point(local_mins.at(i),0),
cv::Point(local_mins.at(i),image.cols),
color);
}
}
int* getVertHist(Mat binImage)
{
int* histArr = new int[binImage.cols];
for (int j = 0; j < binImage.cols; j++)
{
histArr[j] = 0;
for (int i = 0; i < binImage.rows; i++)
{
histArr[j] += binImage.at<uchar>(i,j*3) / 255;
}
}
return histArr;
}
void widenFrame(Mat &frame,int pixels)
{ 
Mat expanded_frame(frame.rows+pixels*2,frame.cols+pixels*2,frame.type());

Mat imageROI = expanded_frame(Rect(pixels,pixels,frame.cols,frame.rows));

frame.copyTo(imageROI);
expanded_frame.copyTo(frame);
}
cv::Mat cutCharWhite(cv::Mat charImage,cv::Mat binImage)
{
if (whiteRatio(binImage)<0.001)
{
return charImage;
}

int line_sum, up_otst = -1,down_otst = -1,left_otst = -1; 

//// вычисление вернего отступа
for (int i = 0; i < binImage.rows; i++)
{
line_sum = 0;
for (int j = 0; j < binImage.cols * binImage.channels(); j++)
{
line_sum += binImage.at<uchar>(i,j);
}

if (line_sum != 0)
{
up_otst = i;
break;
}
}

if (up_otst == -1) // если до конца так и не появилось НЕчерной строчки...
return charImage;


//// вычисление нижнего отступа
for (int i = 0; i < binImage.rows; i++)
{
line_sum = 0;
for (int j = 0; j < binImage.cols * binImage.channels(); j++)
{
line_sum += binImage.at<uchar>(binImage.rows-i-1,j);
}

if (line_sum != 0)
{
down_otst = binImage.rows - i - 1;
break;
}
}


for (int i = 0; i < binImage.cols; i++)
{
line_sum = 0;
for (int j = 0; j < binImage.rows; j++)
{
line_sum += binImage.at<uchar>(j,i);
}

if (line_sum != 0)
{
left_otst = i;
break;
}
}

if (left_otst <= 0)
left_otst = 0;

left_otst = left_otst / 3;
//cv::Mat cut_image=charImage(Rect(0,up_otst,charImage.cols,down_otst - up_otst + 1));
cv::Mat cut_image=charImage(Rect(left_otst,up_otst,charImage.cols-left_otst,down_otst - up_otst + 1));

return cut_image;
//return charImage;
}


// Listener-ы для изменения настроек
void on_trackbar_listener( int, void* )
{
capture.set(CV_CAP_PROP_POS_FRAMES,VIDEO_TRACKBAR_VALUE);
num_frame = VIDEO_TRACKBAR_VALUE;
}
void on_trackbar2_listener( int, void* )
{
CORNERS_THRESH = FASTTHRESH_TRACKBAR_VALUE;
}
void on_trackbar3_listener( int, void* )
{
RADIUS_X = RADIUS_X_TRACKBAR_VALUE;
RADIUS_Y = RADIUS_Y_TRACKBAR_VALUE;
if (COMPR_RATE_TRACKBAR_VALUE == 0)
{
COMPR_RATE = 0.5;
}
else 
{
COMPR_RATE = (float)COMPR_RATE_TRACKBAR_VALUE;
}
}
void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
if( event == CV_EVENT_LBUTTONDOWN )
{
if (borders_setting == 1)
{
if (y > DOWN_BORDER)
{
UP_BORDER = DOWN_BORDER;
DOWN_BORDER = y;
}
else
{UP_BORDER = y;}
borders_setting = 2;
printf("UP_BORDER SET: %d\n",y);
}
else if (borders_setting == 2)
{
if (y < UP_BORDER)
{
DOWN_BORDER = UP_BORDER;
UP_BORDER = y;
}
else
{	DOWN_BORDER = y; }

borders_setting = 0;
printf("DOWN_BORDER SET: %d\n",y);
}
}
}


// Простейшие функции.
char* getVideoTrack()
{
	int i;
	printf("SELECT VIDEO TRACK:\n");
	printf("PRESS '0' FOR _3.avi\n");
	printf("PRESS '1' FOR _4.avi\n");
	printf("PRESS '2' FOR 4 mins.avi\n");
	printf("PRESS '3' FOR all.avi\n");
	printf("PRESS '4' FOR cisterns.avi\n");
	printf("PRESS '5' FOR cisterns2.avi\n");
	printf("PRESS '6' FOR cisterns3.avi\n");
	printf("PRESS '7' FOR hoppers.avi\n");
	printf("PRESS '8' FOR wagons.avi\n");
	printf("PRESS '9' FOR wagons2.avi\n");
	printf("PRESS '10' FOR 0.avi\n");
	printf("PRESS '11' FOR 1.avi\n");
	printf("PRESS '12' FOR 3.avi\n");
	printf("PRESS '13' FOR 4.avi\n");
	printf("PRESS '14' FOR 5.avi\n");
	printf("MY CHOICE: ");
	scanf("%d",&i);
	switch (i)
	{
		case 0:
		NUMBER_HEIGHT = 23;
		return "C:\\Main\\WORK\\video\\_3.avi";
		break;
		case 1:
		NUMBER_HEIGHT = 20;
		return "C:\\Main\\WORK\\video\\_4.avi";
		break;
		case 2:
		NUMBER_HEIGHT = 25; // ???
		return "C:\\Main\\WORK\\video\\4 mins.avi";
		break;
		case 3:
		NUMBER_HEIGHT = 30;
		return "C:\\Main\\WORK\\video\\all.avi";
		break;
		case 4:
		NUMBER_HEIGHT = 38;
		return "C:\\Main\\WORK\\video\\cisterns.avi";
		break;
		case 5:
		NUMBER_HEIGHT = 34;
		return "C:\\Main\\WORK\\video\\cisterns2.avi";
		break;
		case 6:
		NUMBER_HEIGHT = 30;
		CORNERS_THRESH = 25;
		FASTTHRESH_TRACKBAR_VALUE = CORNERS_THRESH;
		return "C:\\Main\\WORK\\video\\cisterns3.avi";
		break;
		case 7:
		NUMBER_HEIGHT = 43;
		return "C:\\Main\\WORK\\video\\hoppers.avi";
		break;
		case 8:
		NUMBER_HEIGHT = 41;
		return "C:\\Main\\WORK\\video\\wagons.avi";
		break;
		case 9:
		NUMBER_HEIGHT = 30;
		return "C:\\Main\\WORK\\video\\wagons2.avi";
		break;
		case 10:
		NUMBER_HEIGHT = 30;
		return "C:\\Main\\WORK\\video\\0.avi";
		break;
		case 11:
		NUMBER_HEIGHT = 25;
		return "C:\\Main\\WORK\\video\\1.avi";
		break;
		case 12:
		NUMBER_HEIGHT = 30;
		return "C:\\Main\\WORK\\video\\3.avi";
		break;
		case 13:
		NUMBER_HEIGHT = 30;
		return "C:\\Main\\WORK\\video\\4.avi";
		break;
		case 14:
		NUMBER_HEIGHT = 30;
		return "C:\\Main\\WORK\\video\\5.avi";
		break;
		default:
		return "C:\\Main\\WORK\\video\\cisterns.avi";
	}
}
cv::Mat binarize(cv::Mat image)
{
cv::Mat grey;
if (image.channels() == 3)
{
cv::cvtColor(image,grey,CV_RGB2GRAY);
}
else image.copyTo(grey);
cv::Mat binary;
cv::threshold(grey,binary,128,255,CV_THRESH_OTSU); // or CV_THRESH_BINARY
if (image.channels() == 3)
{
for (int i = 0; i < image.rows; i++)
for (int j = 0; j < image.cols; j++)
{
image.at<Vec3b>(i,j)[0] = binary.at<uchar>(i,j);
image.at<Vec3b>(i,j)[1] = binary.at<uchar>(i,j);
image.at<Vec3b>(i,j)[2] = binary.at<uchar>(i,j);
}
return image;
}
return binary;
}
float whiteRatio(cv::Mat image)
{
if (image.cols*image.rows == 0) return 0;

int sum = 0;
image = binarize(image);
for (int i = 0; i < image.cols*image.channels(); i++)
for (int j = 0; j < image.rows; j++)
{
sum += image.at<uchar>(j,i);
}
return sum / (image.cols * image.channels() * image.rows);
}
int* medFiltHist(int* arr,int n)
{
// Усредняет значения (сглаживает), проходясь окном шириной = 5.
for (int i = 2; i < n - 2; i++)
{
arr[i] = ( arr[i-2] + arr[i-1] + arr[i] + arr[i+1] + arr[i+2] ) / 5; 
}
return arr;
}
int* calcHist(cv::Mat image,int flag) // flag = 0 - hor, = 1 - vert
{
// Compute histogram
int* histArr; 
//printf("rows=%d; cols=%d;\n",image.rows,image.cols);
if (flag == 0) // горизонтальная гистограмма
{
histArr = new int[image.rows];
for (int j = 0; j < image.rows; j++)
{
histArr[j] = 0;
for (int i = 0; i < image.cols; i++)
{
histArr[j] += image.at<uchar>(j,i);
}
//printf("histArr[%d]=%d;\n",j,histArr[j]);
}
}
else if (flag == 1)
{
histArr = new int[image.cols];
for (int j = 0; j < image.cols; j++)
{
histArr[j] = 0;
for (int i = 0; i < image.rows; i++)
{
histArr[j] += image.at<uchar>(i,j);
}
}
}

return histArr;
}
int* min(int* arr,int n)
{ // Возвращает минимальное значение массива и его порядковый номер в массиве.
int* p = new int[2];
int minVal = 500000;
int minPos = -1;
for (int i = 0; i < n; i++)
{
if (arr[i] <= minVal)
{
minVal = arr[i];
minPos = i;
}
}
p[0] = minVal;
p[1] = minPos;
return p;
}
int* max(int* arr,int n)
{ // Возвращает максимальное значение массива и его порядковый номер в массиве.
int* p = new int[2];
int maxVal = -500000;
int maxPos = -1;
for (int i = 0; i < n; i++)
{
if (arr[i] > maxVal)
{
maxVal = arr[i];
maxPos = i;
}
}
p[0] = maxVal;
p[1] = maxPos;
return p;
}
void printArray(int* arr,int n)
{
for (int i = 0; i < n; i++)
{
printf("arr[%d]=%d;\n",i,arr[i]);
}
}
void prewittFilter(const cv::Mat &inp_image,cv::Mat &out_image,int type)
{
// Construct kernel (initialize all entries with 0)
cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
// assign kernel values 
switch (type){
case 0:
kernel.at<float>(0,0) = 1.0; //horisontal Prewitt filter http://en.wikipedia.org/wiki/Prewitt_operator
kernel.at<float>(0,1) = 1.0;
kernel.at<float>(0,2) = 1.0;
kernel.at<float>(2,0) = -1.0;
kernel.at<float>(2,1) = -1.0;
kernel.at<float>(2,2) = -1.0;
break;
case 1:
kernel.at<float>(0,0) = 1.0; // vertical Prewitt filter
kernel.at<float>(1,0) = 1.0;
kernel.at<float>(2,0) = 1.0;
kernel.at<float>(0,2) = -1.0;
kernel.at<float>(1,2) = -1.0;
kernel.at<float>(2,2) = -1.0;
break;
}
//sharpening image
/*kernel.at<float>(1,1) = 5.0;
kernel.at<float>(1,0) = -1.0;
kernel.at<float>(1,2) = -1.0;
kernel.at<float>(2,1) = -1.0;
kernel.at<float>(0,1) = -1.0;*/

cv::filter2D(inp_image,out_image,inp_image.depth(),kernel);
}
cv::Mat makeBlack(cv::Mat image)
{
for (int i = 0; i < image.rows; i++)
{
for (int j = 0; j < image.cols*image.channels(); j++)
{
image.at<uchar>(i,j) = 0;
}
}
return image;
}
int mean(int* arr,int n)
{
int sum = 0;
for (int i = 0; i < n; i++)
{
sum+=arr[i];
}
return (int)(sum/n);
}
int* cleanHist(int* inp_arr,int n, int thresh)
{
for (int i = 0; i < n; i++)
{
if (inp_arr[i] <= thresh)
inp_arr[i] = 0;
}
return inp_arr;
}
bool belongs(int value, int* arr,int n)
{
for (int i = 0; i < n; i++)
{
if (arr[i] == value)
return true;
}
return false;
}
int* mysort(int* arr,int n)
{
int tmp;
for (int i = 0; i < n; i++)
{
int minval = arr[i];
int minpos = i;
for (int j = i; j < n; j++)
{
if (arr[j] < minval)
{
minval = arr[j];
minpos = j;
}
}
tmp = arr[i];
arr[i] = minval;
arr[minpos] = tmp;
}
return arr;
}
bool belongsToTriangle(cv::Point2f point,std::vector<cv::Point2f> vertices)
{ //принадлежность точки треугольнику
if (vertices.size() == 3)
{
double t1 = (vertices.at(0).x - point.x)*(vertices.at(1).y - vertices.at(0).y) - (vertices.at(1).x - vertices.at(0).x)*(vertices.at(0).y - point.y);
double t2 = (vertices.at(1).x - point.x)*(vertices.at(2).y - vertices.at(1).y) - (vertices.at(2).x - vertices.at(1).x)*(vertices.at(1).y - point.y);
double t3 = (vertices.at(2).x - point.x)*(vertices.at(0).y - vertices.at(2).y) - (vertices.at(0).x - vertices.at(2).x)*(vertices.at(2).y - point.y);
if ((t1 > 0 && t2 > 0 && t3 > 0)||(t1 < 0 && t2 < 0 && t3 < 0))
return true;
}
return false;
}
bool belongsToArea(cv::Point2f point,std::vector<cv::Point2f> vertices)
{// Проверить, принадлежит ли точка области с вершинами (пока только треугольник или четырехугольник)
if (vertices.size() >= 3 && vertices.size() <= 4)
{
if (vertices.size() == 3)
{
return belongsToTriangle(point,vertices);
}
else
{
// Разобьем четырехугольник на два треугольника
std::vector<cv::Point2f> vertices1,vertices2;
vertices1.push_back(vertices.at(0));
vertices1.push_back(vertices.at(1));
vertices1.push_back(vertices.at(2));
vertices2.push_back(vertices.at(0));
vertices2.push_back(vertices.at(2));
vertices2.push_back(vertices.at(3));
return (belongsToTriangle(point,vertices1) || belongsToTriangle(point,vertices2));
}
}
return false;
}
cv::Mat invertImage(cv::Mat image)
{
cv::Mat invImage;
image.copyTo(invImage);
for(int i = 0; i < image.rows; i++)
for(int j = 0; j < image.cols*image.channels(); j++)
        invImage.at<uchar>(i,j) = 255-image.at<uchar>(i,j);
return invImage;
}



// WRITE TO FILE
/*char filename[30];
std::sprintf(filename,"C:\\Main\\WORK\\!test\\%d.bmp",num_frame);
cv::imwrite(filename,image1);*/

