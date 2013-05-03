#include "main.h"

using namespace std;
using namespace cv;

int main()
{
	int trackNum = chooseTrackNum();
	string path = getPathToVideo(trackNum);
	if ( path.length() == 0 )
	{
		return -1;
	}
	
	cout << path << endl;

	return 0;
}