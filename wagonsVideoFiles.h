#ifndef _WAGONS_VIDEONAMES_H_
#define _WAGONS_VIDEONAMES_H_

#include <iostream>
using namespace std;

#define VIDEO_PATH "C:\\Users\\Igor\\Dropbox\\WORK\\Video\\WAGONS_VIDEO\\"

#define FILES_COUNT 11

static const char *VIDEOFILES[FILES_COUNT] = { "_3.avi", 
												"_4.avi",
												"4 mins.avi",
												"all.avi",
												"cisterns.avi",
												"cisterns2.avi",
												"cisterns3.avi",
												"hoppers.avi",
												"wagons.avi",
												"wagons2.avi",
												"stirol1.avi" };
#define DEFAULT_TRACK 5

int chooseVideoTrackNum();
string getPathToVideo(int videoTrackNum);


#endif