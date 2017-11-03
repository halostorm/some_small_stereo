#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <ctime>

using namespace std;
using namespace cv;

int main() {

	VideoCapture camleft(0);
	VideoCapture camright(1);
	
    camleft.set(CV_CAP_PROP_FRAME_WIDTH,640);
    camleft.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	
    camright.set(CV_CAP_PROP_FRAME_WIDTH,640);
    camright.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	

	Mat img_left, img_right;

	int num = 0;
	char imageleft[50];
	char imageright[50];

    
    

	while (1)
	{
        if(!camleft.isOpened()||!camright.isOpened())
        {
            cout<<"camera open error"<<endl;
            return -1;
        }
        
        cout<<"camera open succeed "<<endl;
        
        
		camleft   >> img_left;
		camright  >> img_right;

		imshow("camleft",img_left);
		imshow("camright",img_right);
        

		char c = waitKey(10);
		if (c == 27)
		{
            //按esc退出
			break;
		}
		if (c == 32)
		{
            //拍空格保存
            
			cout<<num<<endl;
            
            time_t now = time(0);
            tm* ltm = localtime(&now);
            
			sprintf(imageleft,  "%s%02d%02d%02d%02d%02d%s","../camleft/camleft_", 1+ltm->tm_mon,ltm->tm_mday,ltm->tm_hour,ltm->tm_min,ltm->tm_sec,".png");
            sprintf(imageright,  "%s%02d%02d%02d%02d%02d%s","../camright/camright_", 1+ltm->tm_mon,ltm->tm_mday,ltm->tm_hour,ltm->tm_min,ltm->tm_sec,".png");
			
            
			imwrite(imageleft, img_left);
			imwrite(imageright, img_right);	
            
			num++;	
		}
	}
	return 0;
	
}
