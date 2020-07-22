#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include  "Harris.h"

using namespace cv;
using namespace std;

//检测并标记角点的Harris 2种方式

int Harris()
{
	Mat img = imread("bdb2.bmp");
	imshow("src", img);
	Mat result = img.clone();
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat dst;
	Mat corner_img;
	cornerHarris(gray, corner_img, 3, 23, 0.04);
	imshow("corner", corner_img);

	threshold(corner_img, dst, 0.005, 255, THRESH_BINARY);
	imshow("dst", dst);

	int rowNumber = gray.rows;  //获取行数
	int colNumber = gray.cols;  //获取每一行的元素
	cout << rowNumber << endl;
	cout << colNumber << endl;
	cout << dst.type() << endl;

	for (int i = 1; i < rowNumber - 1; i++)
	{
		for (int j = 1; j < colNumber - 1; j++)
		{
			if (dst.at<float>(i, j) == 255)
			{
				circle(result, Point(j, i), 5, Scalar(0, 0, 255), 2, 8);
			}
		}
	}

	imshow("result", result);
	//waitKey();
	//cv::Mat  image, image1 = cv::imread("1.jpg");    //灰度变换
	//cv::cvtColor (image1,image,COLOR_BGR2GRAY);
	//// 经典的harris角点方法
	//harris harris;
	//// 计算角点
	//harris.detect(image);
	////获得角点
	//std::vector<cv::Point> pts;
	//harris.getCorners(pts,0.01);
	//// 标记角点
	//harris.drawOnImage(image,pts);
	//cv::namedWindow ("harris");
	//cv::imshow ("harris",image);
	cv::waitKey(0);

	return 0;
}