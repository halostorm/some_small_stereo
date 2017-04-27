#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
StereoSGBM sgbm;


void uptatesgbm()
{
	sgbm.SADWindowSize = 3;
	sgbm.preFilterCap = 63;
	sgbm.P1 = 8 *3* sgbm.SADWindowSize* sgbm.SADWindowSize;
	sgbm.P2 = 32* 3 * sgbm.SADWindowSize * sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = 320;
	sgbm.uniquenessRatio = 20;
	sgbm.speckleRange = 32;
	sgbm.speckleWindowSize = 100;
	sgbm.disp12MaxDiff = 0;
	sgbm.fullDP = 0;
}

int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)
{
      // 将原始视差数据的位深转换为 8 位
	Mat disp8u;
	if (disparity.depth() != CV_8U)
	{
	  if (disparity.depth() == CV_8S)
	  {
	    disparity.convertTo(disp8u, CV_8U);
	    
	  }
	  else
	  {
	      double minVal; double maxVal;
	      minMaxLoc( disparity, &minVal, &maxVal );
		disparity.convertTo(disp8u, CV_8UC1, 255 / (maxVal - minVal));
	    }
	   }
	   else
	   {
		disp8u = disparity;
	    }

    // 转换为伪彩色图像 或 灰度图像
    if (isColor)
    {
        if (disparityImage.empty() || disparityImage.type() != CV_8UC3 || disparityImage.size() != disparity.size())
        {
            disparityImage = cv::Mat::zeros(disparity.rows, disparity.cols, CV_8UC3);
        }

        for (int y = 0; y<disparity.rows; y++)
        {
            for (int x = 0; x<disparity.cols; x++)
            {
                uchar val = disp8u.at<uchar>(y, x);
                uchar r, g, b;

                if (val == 0)
                    r = g = b = 0;
                else
                {
                    r = 255 - val;
                    g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
                    b = val;
                }

                disparityImage.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);

            }
        }
    }
    else
    {
        disp8u.copyTo(disparityImage);
    }

    return 1;
}


int main(int argc, char** argv)
{
	Mat left,right;
	
	string img1=argv[1];
	string img2=argv[2];
	

	cout<<img1<<endl<<img2<<endl;
		
	left = imread( img1, CV_LOAD_IMAGE_COLOR);
	right = imread( img2, CV_LOAD_IMAGE_COLOR);
	
	Size imagesize(left.cols,left.rows);
	
	Mat img(imagesize.height, imagesize.width * 2, CV_8UC3);//高度一样，宽度双倍
	Mat imgPart1 = img( Rect(0, 0, imagesize.width, imagesize.height) );//浅拷贝
	Mat imgPart2 = img( Rect(imagesize.width, 0, imagesize.width, imagesize.height) );//浅拷贝
	resize(left, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
	resize(right, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);
	
	for( int i = 0; i < img.rows; i += 64 )
	    line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);

	//显示行对准的图形
	Mat smallImg;//由于分辨率1:1显示太大，所以缩小显示
	resize(img, smallImg, Size(), 0.5, 0.5, CV_INTER_AREA);
	imshow("rectified", smallImg);
	
	uptatesgbm();
	
	Mat disp,dispcolor,disp_small;
	sgbm(left, right, disp);
	
	getDisparityImage(disp,dispcolor,true);
	resize(dispcolor, disp_small, Size(), 0.5, 0.5, CV_INTER_AREA);
	
	imshow("disp", disp_small);
	waitKey(0);
	
	return 0;
}
