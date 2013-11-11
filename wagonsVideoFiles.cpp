#include <iostream>
#include <stdio.h>
#include "wagonsVideoFiles.h"

using namespace std;

int chooseVideoTrackNum()
{
	int trackNum = -1;

	cout << "Available video files:" << endl;
	for (int i = 1; i <= FILES_COUNT; i++) {
		cout << "Track " << i << ": " << VIDEOFILES[i-1] << endl;
	}
	
	cout << "Enter valid track number: ";
	cin >> trackNum;
	
	return trackNum;
}

string getPathToVideo(int trackNum)
{
	if ( trackNum <= 0 || trackNum > FILES_COUNT )
	{
		cout << "Incorrect file number!" << endl;
		cout << VIDEOFILES[DEFAULT_TRACK] << " was selected!" << endl;
	}

	string pathToVideo = VIDEO_PATH;

	pathToVideo.append(VIDEOFILES[trackNum-1]);

	return pathToVideo;
}
