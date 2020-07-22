#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

//精确定位角点坐标。
int ShiTomasi()
{
	Mat srcImage = imread("bdb2.bmp");
	if (srcImage.empty())
	{
		printf("could not load image..\n");
		return false;
	}
	Mat srcgray, dstImage, normImage, scaledImage;
	cvtColor(srcImage, srcgray, COLOR_BGR2GRAY);
	Mat srcbinary;
	threshold(srcgray, srcbinary, 0, 255, THRESH_OTSU | THRESH_BINARY);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
	morphologyEx(srcbinary, srcbinary, MORPH_OPEN, kernel, Point(-1, -1));

	vector<Point2f> corners;
	//提供初始角点的坐标位置和精确的坐标的位置
	int maxcorners = 200;
	double qualityLevel = 0.01;
	//角点检测可接受的最小特征值
	double minDistance = 10;	//角点之间最小距离
	int blockSize = 3;//计算导数自相关矩阵时指定的领域范围
	double  k = 0.04; //权重系数
	goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);	//Mat():表示感兴趣区域；false:表示不用Harris角点检测 	//输出角点信息
	cout << "角点信息为：" << corners.size() << endl; 	//绘制角点
	RNG rng(12345);
	for (unsigned i = 0; i < corners.size(); i++)
	{
		circle(srcImage, corners[i], 2, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
		cout << "角点坐标：" << corners[i] << endl;
	}

	return 0;
}
//————————————————
//版权声明：本文为CSDN博主「令仪.雅」的原创文章
//原文链接：https ://blog.csdn.net/xinyuski/article/details/93472253