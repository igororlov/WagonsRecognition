#ifndef _WAGONS_VIDEONAMES_H_
#define _WAGONS_VIDEONAMES_H_

#include <iostream>
using namespace std;

#define VIDEO_PATH "C:\\Users\\Igor\\Dropbox\\WORK\\Video\\WAGONS_VIDEO\\"

#define FILES_COUNT 10

#define FILE_1 "_3.avi"
#define FILE_2 "_4.avi"
#define FILE_3 "4 mins.avi"
#define FILE_4 "all.avi"
#define FILE_5 "cisterns.avi"
#define FILE_6 "cisterns2.avi"
#define FILE_7 "cisterns3.avi"
#define FILE_8 "hoppers.avi"
#define FILE_9 "wagons.avi"
#define FILE_10 "wagons2.avi"

int chooseTrackNum();
string getPathToVideo(int trackNum);


#endif