#include <iostream>
#include <stdio.h>
#include "wagonsVideoFiles.h"

using namespace std;

int chooseVideoTrackNum()
{
	int trackNum = -1;

	cout << "Available video files:" << endl;
	cout << "Track 1: " << FILE_1 << endl;
	cout << "Track 2: " << FILE_2 << endl;
	cout << "Track 3: " << FILE_3 << endl;
	cout << "Track 4: " << FILE_4 << endl;
	cout << "Track 5: " << FILE_5 << endl;
	cout << "Track 6: " << FILE_6 << endl;
	cout << "Track 7: " << FILE_7 << endl;
	cout << "Track 8: " << FILE_8 << endl;
	cout << "Track 9: " << FILE_9 << endl;
	cout << "Track 10: " << FILE_10 << endl;
	
	cout << "Enter valid track number: ";
	cin >> trackNum;
	
	return trackNum;
}

string getPathToVideo(int trackNum)
{
	if ( trackNum <= 0 || trackNum > FILES_COUNT )
	{
		cout << "Incorrect file number!" << endl;
		cout << DEFAULT_TRACK << " was selected!" << endl;
	}

	string pathToVideo = VIDEO_PATH;
	switch ( trackNum ) 
	{
	case 1:
		pathToVideo.append(FILE_1);
		break;
	case 2:
		pathToVideo.append(FILE_2);
		break;
	case 3:
		pathToVideo.append(FILE_3);
		break;
	case 4:
		pathToVideo.append(FILE_4);
		break;
	case 5:
		pathToVideo.append(FILE_5);
		break;
	case 6:
		pathToVideo.append(FILE_6);
		break;
	case 7:
		pathToVideo.append(FILE_7);
		break;
	case 8:
		pathToVideo.append(FILE_8);
		break;
	case 9:
		pathToVideo.append(FILE_9);
		break;
	case 10:
		pathToVideo.append(FILE_10);
		break;
	default:
		pathToVideo.append(DEFAULT_TRACK);
		break;
	}
	return pathToVideo;
}
