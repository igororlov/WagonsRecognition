#ifndef _WAGONS_VIDEONAMES_H_
#define _WAGONS_VIDEONAMES_H_

#include <iostream>
using namespace std;

#define VIDEO_PATH "C:\\Users\\Igor\\Dropbox\\WORK\\Video\\WAGONS_VIDEO\\"

#define FILES_COUNT 11

static const char *VIDEOFILES[FILES_COUNT] = { "_3", 
												"_4",
												"4 mins",
												"all",
												"cisterns",
												"cisterns2",
												"cisterns3",
												"hoppers",
												"wagons",
												"wagons2",
												"stirol1" };
#define DEFAULT_TRACK 5
#define VIDEO_EXTENSION ".avi"
#define CONFIG_EXTENSION ".txt"

int chooseVideoTrackNum();
string getPathToVideo(int videoTrackNum);


#endif