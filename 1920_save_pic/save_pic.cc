#include <iostream>
#include<opencv2/opencv.hpp>
#include<cstdio>

using namespace std;
using namespace cv;

int main() {

	VideoCapture camleft(1);
	VideoCapture camright(2);
	
	camleft.set(CV_CAP_PROP_FRAME_WIDTH,1920);
	camleft.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
	
	camright.set(CV_CAP_PROP_FRAME_WIDTH,1920);
	camright.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
	

	Mat img_left, img_right;

	int num= 0;
	char imageleft[30];
	char imageright[30];
	cout << "test"<<endl;
	
	vector<int>compression_params;    
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION); //PNG格式图片的压缩级别    
	compression_params.push_back(0);  
	
	 Mat smallImg;//由于分辨率1:1显示太大，所以缩小显示
    

	while (1)
	{
		camleft >> img_left;
		camright >> img_right;

		resize(img_left, smallImg, Size(), 0.5, 0.5, CV_INTER_AREA);
		imshow("rectified", smallImg);

		char c = waitKey(30);
		if (c == 27)
		{
			break;
		}
		if (c == 32)
		{
			cout<<num<<endl;		
			sprintf(imageleft,  "%s%d%s","../left/lefttest", num, ".png");
			sprintf(imageright, "%s%d%s", "../right/righttest", num, ".png");
			imwrite(imageleft, img_left,compression_params);
			imwrite(imageright, img_right,compression_params);			
			num++;	
		}
	}
	return 0;
	
}
