#include <highgui.h>
#include <cvaux.h>
#include <iostream>



using namespace cv;
using namespace std;


bool left_mouse = false;

Mat image, image1;

const Scalar GREEN = Scalar(0, 255, 0);

int rect_width = 0, rect_height = 0;

int m_frameWidth = 640;
int m_frameHeight = 480;

bool    m_Calib_Data_Loaded;        // 是否成功载入定标参数
cv::Mat m_Calib_Mat_Q;              // Q 矩阵
cv::Mat m_Calib_Mat_Remap_X_L;      // 左视图畸变校正像素坐标映射矩阵 X
cv::Mat m_Calib_Mat_Remap_Y_L;      // 左视图畸变校正像素坐标映射矩阵 Y
cv::Mat m_Calib_Mat_Remap_X_R;      // 右视图畸变校正像素坐标映射矩阵 X
cv::Mat m_Calib_Mat_Remap_Y_R;      // 右视图畸变校正像素坐标映射矩阵 Y
cv::Mat m_Calib_Mat_Mask_Roi;       // 左视图校正后的有效区域
cv::Rect m_Calib_Roi_L;             // 左视图校正后的有效区域矩形
cv::Rect m_Calib_Roi_R;             // 右视图校正后的有效区域矩形
double          m_FL;

int m_numberOfDisparies;            // 视差变化范围


Point origin;
cv::StereoSGBM sgbm;

cv::Mat  disp,   imageLeft, imageRight, disparityImage;



static void onMouse(int event, int x, int y, int flags, void* userdate){


	origin=Point(x,y);
	Mat mouse_show;
    mouse_show = *(Mat *)userdate;
	
    if (event == CV_EVENT_LBUTTONDOWN)
    {
		cout << origin <<"in world coordinate is: " << mouse_show.at<Vec3f>(origin) << endl;     
		cout << disp.channels();
                                                                               
        left_mouse = true;
    }
}

int getPointClouds(Mat& disparity, Mat& pointClouds, cv::Mat m_Calib_Mat_Q )
{
	if (disparity.empty())
	{
		return 0;
	}

	//计算生成三维点云
	cv::reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);
    pointClouds *= 1.6;
	
	// 校正 Y 方向数据，正负反转
	// 原理参见：http://blog.csdn.net/chenyusiyuan/article/details/5970799 
	for (int y = 0; y < pointClouds.rows; ++y)
	{
		for (int x = 0; x < pointClouds.cols; ++x)
		{
			cv::Point3f point = pointClouds.at<cv::Point3f>(y,x);
            //point.y = -point.y;
			point.z = -point.z;
			pointClouds.at<cv::Point3f>(y,x) = point;
		}
	}

	return 1;
}


int loadCalibData()
{
    // 读入摄像头定标参数 Q roi1 roi2 mapx1 mapy1 mapx2 mapy2
    try
    {
        cv::FileStorage fs("../calib_paras.xml", cv::FileStorage::READ);
        cout << fs.isOpened() << endl;

        if (!fs.isOpened())
        {
            return (0);
        }

        cv::Size imageSize;
        cv::FileNodeIterator it = fs["imageSize"].begin();

        it >> imageSize.width >> imageSize.height;
    //  if (imageSize.width != m_frameWidth || imageSize.height != m_frameHeight)   {           return (-1);        }

        vector<int> roiVal1;
        vector<int> roiVal2;

        fs["leftValidArea"] >> roiVal1;

        m_Calib_Roi_L.x = roiVal1[0];
        m_Calib_Roi_L.y = roiVal1[1];
        m_Calib_Roi_L.width = roiVal1[2];
        m_Calib_Roi_L.height = roiVal1[3];

        fs["rightValidArea"] >> roiVal2;
        m_Calib_Roi_R.x = roiVal2[0];
        m_Calib_Roi_R.y = roiVal2[1];
        m_Calib_Roi_R.width = roiVal2[2];
        m_Calib_Roi_R.height = roiVal2[3];


        fs["QMatrix"] >> m_Calib_Mat_Q;
        fs["remapX1"] >> m_Calib_Mat_Remap_X_L;
        fs["remapY1"] >> m_Calib_Mat_Remap_Y_L;
        fs["remapX2"] >> m_Calib_Mat_Remap_X_R;
        fs["remapY2"] >> m_Calib_Mat_Remap_Y_R;

        cv::Mat lfCamMat;
        fs["leftCameraMatrix"] >> lfCamMat;
        m_FL = lfCamMat.at<double>(0, 0);

        m_Calib_Mat_Q.at<double>(3, 2) = -m_Calib_Mat_Q.at<double>(3, 2);

        m_Calib_Mat_Mask_Roi = cv::Mat::zeros(m_frameHeight, m_frameWidth, CV_8UC1);
        cv::rectangle(m_Calib_Mat_Mask_Roi, m_Calib_Roi_L, cv::Scalar(255), -1);

 

        m_Calib_Data_Loaded = true;

        string method;
        fs["rectifyMethod"] >> method;
        if (method != "BOUGUET")
        {
            return (-2);
        }

    }
    catch (std::exception& e)
    {
        m_Calib_Data_Loaded = false;
        return (-99);
    }

    return 1;


}

void uptatesgbm()
{
	sgbm.SADWindowSize = 3;
	sgbm.preFilterCap = 63;
	sgbm.P1 = 8 *3* sgbm.SADWindowSize* sgbm.SADWindowSize;
	sgbm.P2 = 32 *3* sgbm.SADWindowSize* sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = 128;
	sgbm.uniquenessRatio = 15;
	sgbm.speckleRange = 32;
	sgbm.speckleWindowSize = 100;
	sgbm.disp12MaxDiff = 1;
	sgbm.fullDP = 1;
}

int sgbmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity, cv::Mat& imageLeft, cv::Mat& imageRight)
{
	// 输入检查
	if (frameLeft.empty() || frameRight.empty())
	{
		disparity = cv::Scalar(0);
		return 0;
	}
	if (m_frameWidth == 0 || m_frameHeight == 0)
	{

			return 0;
	}

	// 复制图像
	cv::Mat img1proc, img2proc;
	frameLeft.copyTo(img1proc);
	frameRight.copyTo(img2proc);

	// 校正图像，使左右视图行对齐	
	cv::Mat img1remap, img2remap;
	if (m_Calib_Data_Loaded)
	{
		remap(img1proc, img1remap, m_Calib_Mat_Remap_X_L, m_Calib_Mat_Remap_Y_L, cv::INTER_LINEAR);		// 对用于视差计算的画面进行校正
		remap(img2proc, img2remap, m_Calib_Mat_Remap_X_R, m_Calib_Mat_Remap_Y_R, cv::INTER_LINEAR);
	} 
	else
	{
		img1remap = img1proc;
		img2remap = img2proc;
	}

	// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	cv::Mat img1border, img2border;
	if (m_numberOfDisparies != sgbm.numberOfDisparities)
		m_numberOfDisparies = sgbm.numberOfDisparities;
	copyMakeBorder(img1remap, img1border, 0, 0, sgbm.numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(img2remap, img2border, 0, 0, sgbm.numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	// 计算视差
	cv::Mat dispBorder;
	sgbm(img1border, img2border, dispBorder);

	// 截取与原始画面对应的视差区域（舍去加宽的部分）
	cv::Mat disp;
	disp = dispBorder.colRange(sgbm.numberOfDisparities, img1border.cols);	
	disp.copyTo(disparity, m_Calib_Mat_Mask_Roi);

	// 输出处理后的图像
	imageLeft = img1remap.clone();
	imageRight = img2remap.clone();
	rectangle(imageLeft, m_Calib_Roi_L, CV_RGB(0,255,0), 3);
	rectangle(imageRight, m_Calib_Roi_R, CV_RGB(0,255,0), 3);

	return 1;
}



int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)
{
    // 将原始视差数据的位深转换为 8 位
    cv::Mat disp8u;
    if (disparity.depth() != CV_8U)
    {
        if (disparity.depth() == CV_8S)
        {
            disparity.convertTo(disp8u, CV_8U);
        }
        else
        {
            disparity.convertTo(disp8u, CV_8U, 255 / (m_numberOfDisparies*16.));
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
    //读取摄像头
    VideoCapture cap(1); 
    VideoCapture cap1(2);
	
	cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

	cap1.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	
    if (!cap.isOpened())
    {
        cout << "error happened while open cam 1"<<endl;
        return -1;
    }
    if (!cap1.isOpened())
    {
        cout << "error happened while open cam 2" << endl;
        return -1;
    }

   // namedWindow("left", 1);
   // namedWindow("right", 1);
   // namedWindow("dispcolor", 1);
    loadCalibData();

    cout << m_Calib_Data_Loaded << endl;
	
	Size imagesize(640,480);

    while (true)
    {

            Mat frame;
            Mat frame1;
            cap.read(frame);
            cap1.read(frame1);
            if (frame.empty())          break;
            if (frame1.empty())         break;

            frame.copyTo(image);
            frame1.copyTo(image1);
            uptatesgbm();
            sgbmMatch(frame, frame1, disp, imageLeft, imageRight);
            //imshow("left", imageLeft);
            //imshow("right", imageRight);
			
			Mat img(imagesize.height, imagesize.width * 2, CV_8UC3);//高度一样，宽度双倍
			Mat imgPart1 = img( Rect(0, 0, imagesize.width, imagesize.height) );//浅拷贝
			Mat imgPart2 = img( Rect(imagesize.width, 0, imagesize.width, imagesize.height) );//浅拷贝
			resize(imageLeft, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
			resize(imageRight, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);
		
		for( int i = 0; i < img.rows; i += 64 )
			line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);

		//显示行对准的图形
		Mat smallImg;//由于分辨率1:1显示太大，所以缩小显示
		resize(img, smallImg, Size(), 0.8, 0.8, CV_INTER_AREA);
		imshow("rectified", smallImg);
			
			//get_true_disp(disp);
            getDisparityImage(disp, disparityImage, true);
			Mat disp1,cloud;
			//disp.convertTo(disp1, CV_32F, 1.0/16);
			
			getPointClouds(disp, cloud, m_Calib_Mat_Q);
			setMouseCallback("dispcolor", onMouse, &cloud);
			imshow("dispcolor", disparityImage);
			
		
            char c= waitKey(30);
			if(c == 27)
				break;
			if(c == 32)
			{
				imwrite("left1.png",imageLeft);
				imwrite("right1.png",imageRight);
			}


    }
	   return 0;

}
