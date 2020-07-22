#include <opencv2/opencv.hpp>
#include "putTextZH.h"
#include  <iostream>
#include  <stdio.h>
#include  "debug.h"
//#include "putTextZH.cpp"
#define   PI 3.1415

using namespace std;
using namespace cv;

void fastIntegral(unsigned char* inputMatrix, unsigned long* outputMatrix, int width, int height)
{//快速积分图
	unsigned long* columnSum = new unsigned long[width]; // sum of each column
	// calculate integral of the first line
	for (int i = 0; i < width; i++) {
		columnSum[i] = inputMatrix[i];
		outputMatrix[i] = inputMatrix[i];
		if (i > 0) {
			outputMatrix[i] += outputMatrix[i - 1];
		}
	}
	for (int i = 1; i < height; i++) {
		int offset = i * width;
		// first column of each line
		columnSum[0] += inputMatrix[offset];
		outputMatrix[offset] = columnSum[0];
		// other columns
		for (int j = 1; j < width; j++) {
			columnSum[j] += inputMatrix[offset + j];
			outputMatrix[offset + j] = outputMatrix[offset + j - 1] + columnSum[j];
		}
	}
	return;
}
int main()
{
	Mat src = imread("C://Users//xjh000//Desktop//test2.jpg");
#if mask1
	Size size(500, 500);

	cv::resize(src, src, size);

	Mat dst_Img;
	Mat dst = Mat(size, src.type());
	Mat kernel = (Mat_<uchar>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	resize(src, dst_Img, size);

	namedWindow("Orignal", 0);
	namedWindow("Mask+", 0);
	filter2D(dst_Img, dst, src.depth()/*-1*/, kernel);

	imshow("Orignal", src);
	imshow("Mask+", dst);
	waitKey(900);
	destroyAllWindows();
#endif	//mask1
#if mask2
	Rect my_ROI(100, 50, 2000, 2000);
	Mat img = src.clone();
	Mat img1;
	Mat img2;
	Mat mask = Mat(src.size(), src.type());
	cv::namedWindow("1", 0);
	cv::namedWindow("2", 0);

	img1 = img(my_ROI);
	mask(my_ROI).setTo(0);
	img.copyTo(img2, mask);

	cv::imshow("1", img1);
	cv::imshow("2", img2);
	waitKey(0);
#endif  //mask2
#if p_MAT

	if (!src.data)//(src.empty())
	{
		std::cout << "could not load image..." << endl;
	}
	Mat dst = Mat(src.size(), src.type(), Scalar(127, 0, 255));
	//Mat dst=src.clone();
	//src.copeTo(dst);
	//namedWindow("1", 0);
	namedWindow("lk", 0);
	resizeWindow("lk", Size(100, 100));
	//imshow("1", dst);
	cv::waitKey(0);
	cv::cvtColor(dst, dst, COLOR_BGR2GRAY);
	//imshow("2", dst);
	std::cout << src.cols << " " << src.rows << endl;

	Mat M(3888, 3000, CV_8UC1, Scalar(100));
	Mat N;
	N.create(400, 400, CV_8UC3);
	N = Scalar(8, 45, 80);
	//cv::imshow("1", N);
	cv::waitKey(0);

	Mat O = Mat::zeros(3, 3, CV_8UC1);//0
	std::cout << O;

	Mat P = Mat::eye(4, 4, CV_8UC3);//255
	//cv::imshow("2", P);

	Mat array = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	//Mat N3 = imread("test1.jpg");
	Mat N2;
	filter2D(src, N2, -1, array);
	cv::imshow("lk", N2);
	cv::waitKey(0);
#endif //p_MAT
#if pixel_operation
	//只有 src.cols&rows;
	//单通道 <uchar> 多通道 <Vec3b>--<Vec3f> double类型
	//单通道----cvtColor---多通道
	// CV_8UCx-----covertTO------CV_32Fx
	resize(src, src, Size(2000, 2000));
	Mat dst = Mat(src.size(), src.type(), Scalar(0, 0, 255));
	Mat gary = Mat(src.size(), CV_8UC1);
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			int pixel[3];
			pixel[0] = src.at<Vec3b>(row, col)[0];
			pixel[1] = src.at<Vec3b>(row, col)[0];
			pixel[2] = src.at<Vec3b>(row, col)[0];
			//灰度图的处理方式
			gary.at<uchar>(row, col) = max(pixel[0], max(pixel[1], pixel[2]));
			//gary.at<uchar>(row, col) = min(pixel[0], min(pixel[1], pixel[2]));
			dst.at<Vec3b>(row, col)[0] = saturate_cast(255 - pixel[0]);
			dst.at<Vec3b>(row, col)[1] = 255 - pixel[1];
			dst.at<Vec3b>(row, col)[2] = 255 - pixel[2];
		}
	}
	//bitwise_not(src, dst);//取反
	//bitwise_or(src, dst, dst);
	//bitwise_and(src, dst, dst);
	//bitwise_xor(src, dst, dst);

	namedWindow("oppose", 0);
	imshow("oppose", gary);
	waitKey(0);
#endif // pixel_operation
#if image_blending;
	//alpha：addWeighted混合两个同类型，同尺寸的图片
	//dst(I) = saturate(src1(I) * alpha + src2(I)* (1-alpha)+ gamma);
	//add ，multiply，subtract像素直接计算

	Mat src1 = imread("C://Users//xjh000//Desktop//32.jpg");
	resize(src, src, src1.size());
	Mat dst;
	double alpha = 0.5;
	double gamma = 10.0;
	addWeighted(src, alpha, src1, 1.0 - alpha, gamma, dst, -1);
	//subtract(src, src1, dst);
	namedWindow("blending", 0);
	imshow("blending", dst);
	waitKey(0);
#endif // image blending;
#if adjustment
	/*图像变换可以看作如下 :
	像素变换 - 点操作
	邻域操作 - 区域
	调整图像亮度和对比度属于像素变换点操作
	g(i, j) = a*f(i, j) + b；其中a > 0(0<a<1,对比度和亮度降低；反之，亦反), b是增益变量*/
	namedWindow("yuantu", 0);
	imshow("yuantu", src);
	Mat dst(src.size(), src.type());
	double alpha = 1.8, beta = 10;
	for (int rows = 0; rows < src.rows; rows++) {
		for (int cols = 1; cols < src.cols; cols++) {
			Vec3b pixel;
			pixel = src.at<Vec3b>(rows, cols);
			dst.at<Vec3b>(rows, cols)[0] = saturate_cast<uchar>(pixel[0] * alpha + beta);
			dst.at<Vec3b>(rows, cols)[1] = saturate_cast<uchar>(pixel[1] * alpha + beta);
			dst.at<Vec3b>(rows, cols)[2] = saturate_cast<uchar>(pixel[2] * alpha + beta);
		}
	}
	namedWindow("adjust", 0);
	imshow("adjust", dst);
	waitKey(0);
#endif // adjustment
#if drawing
	/*绘制线、矩形、园、椭圆等基本几何形状
		画线line(LINE_ _4\LINE_ _8\LINE_ _AA)
		画椭圆ellipse
		画矩形rectangle
		画圆circle
		填充fillPoly*/
	Point org = Point(2200, 1200);
	putText(src, "I LOVE HEU!", org, FONT_HERSHEY_PLAIN, 10.0, Scalar(255, 0, 0), 20, 8);
	namedWindow("text", 0);
	//imshow("text", src);
	//waitKey(0);
	Point pts[1][5];
	pts[0][0] = Point(1000, 1000);
	pts[0][1] = Point(1000, 2000);
	pts[0][2] = Point(2000, 2000);
	pts[0][3] = Point(2000, 1000);
	pts[0][4] = Point(1000, 1000);

	const Point* ppts[] = { pts[0] };
	int npt[] = { 5 };
	Scalar color = Scalar::all(255);
	fillPoly(src, ppts, npt, 1, color, LINE_AA, 0, Point(800, 1000));
	imshow("text", src);
	waitKey(0);

#endif // drawing
#if dim
	/*模糊原理
	Smooth / Blur是图像处理中最简单和常用的操作之一
	使用该操作的原因之一就为了给图像预处理时候减低噪声
	●使用Smooth/Blur操作其背后是数学的卷积计算
	●通常这些卷积算子计算都是线性操作, 所以又叫线性滤波（mask，kernel）
	比如：1,归一化盒子滤波(均值滤波，权重相等) 2，高斯滤波（正态分布）
		  3,中值（Max）（Min）滤波(减少椒盐噪声，更饱和)	4,高斯双边滤波(像素处理半径，-1，卷积核)
		  ksize,xsymax,ysymax	 奇数且>0
	问题：
	●均值模糊无法克服边缘像素信息丢失缺陷。原因是均值滤波是基于平均权重
	●高斯模糊部分克服 了该缺陷,但是无法完全避免，因为没有考虑像素值的不同
	●高斯双边模糊-是边缘保留的滤波方法,避免了边缘信息丢失,保留了图像轮廓不变*/
	Mat dst(src.size(), src.type());
	//blur(src, dst, Size(1, 29), Point(-1, -1));
	//GaussianBlur(src, dst, Size(15, 3), 1, 4);
	//medianBlur(src, dst, 3);
	//Mat kernel = (Mat_<uchar>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	bilateralFilter(src, dst, 29, -1, 3);
	//filter2D(dst, dst, -1, kernel, Point(-1, -1), 0, 4);
	cv::namedWindow("diming", 0);
	cv::imshow("diming", dst);
	cv::waitKey(0);
#endif // dim
#if eroding_dilating
	/*图像形态学操作 - 基于形状的一系列图像处理操作的合集,
	主要是基于集合论基础上的形态学数学
	形态学有四个基本操作 : 腐蝕、膨胀、开、闭
	膨胀与腐蚀是图像处理中最常用的形态学操作手段*/
	/*膨胀：
	跟卷积操作类似,假设有图像A和结构元素BI结构元
	素B在A上面移动,其中B定义其中心为锚点,计算B覆
	盖下A的最大像素值用来替换锚点的像素,其中B作为
	结构体可以是任意形状*/
	/*腐蚀：
	类似，最小值替换*/
	//绘制结构元素B
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(10, 100), Point(-1, -1));
	//dilate(src, src, kernel);
	erode(src, src, kernel);
	cv::namedWindow("t1", 0);
	cv::imshow("t1", src);
	cv::waitKey(0);

#endif // eroding_dilating
#if morphology
	/*开运算:先腐蚀后膨胀。去掉小物体。
	- int OPT- CV_MOP_OPEN/ CV_ MOP_CLOSE/ CV_ MOP_ GRADIENT
	CV_ MOP_ _TOPHAT/ CV_ MOP_ BLACKHAT 形态学操作类型
	MORPH_ERODE    = 0, //腐蚀
	MORPH_DILATE   = 1, //膨胀
	MORPH_OPEN     = 2, //开操作，深色有小空白
	MORPH_CLOSE    = 3, //闭操作,纯色有小污渍。
	MORPH_GRADIENT = 4, //梯度操作.膨胀-腐蚀
	MORPH_TOPHAT   = 5, //顶帽操作,原图像-开操作
	MORPH_BLACKHAT = 6, //黑帽操作
	MORPH_HITMISS  = 7  */
	/*应用：操作水平线和垂直线
	adaptiveThreshold(
	Mat src, //输入的灰度图像
	Matdest, //二值图像
	double maxValue,//二值图像最大值
	int adaptiveMethod//自适应方法，只能其中之一一
	// ADAPTIVE THRESH MEAN C，ADAPTIVE THRESH_ GAUSSIAN C
	int thresholdType//阈们类型
	int blockSize,//块大小
	doubleC //常量C可以是正数,0,负数*/
	namedWindow("test", 0);
	/*Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(-1, -1));
	morphologyEx(src, src, 5, kernel);*/
	Mat temp, dst, gray;
	Mat src_ = imread("C://Users//xjh000//Desktop//test3.png", 0);

	threshold(src_, gray, 15, 255, ADAPTIVE_THRESH_GAUSSIAN_C);
	imshow("test", src_);
	waitKey(0);
	Mat shui = getStructuringElement(MORPH_RECT, Size(10, 1), Point(-1, -1));
	Mat cui = getStructuringElement(MORPH_RECT, Size(1, 10), Point(-1, -1));

	erode(gray, temp, cui);
	dilate(temp, dst, cui);

	imshow("test", dst);
	waitKey(0);
#endif // morphology
#if coner
	//代码里面有三种程序

	Mat srcImage = imread("C:/Users/zhj/Desktop/image/template.bmp");

	if (srcImage.empty())
	{
		printf("could not load image..\n");
		return false;
	}
	Mat srcgray, dstImage, normImage, scaledImage;

	cvtColor(srcImage, srcgray, CV_BGR2GRAY);

	Mat srcbinary;
	threshold(srcgray, srcbinary, 0, 255, THRESH_OTSU | THRESH_BINARY);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
	morphologyEx(srcbinary, srcbinary, MORPH_OPEN, kernel, Point(-1, -1));

	/*
		//1、Harris角点检测
		cornerHarris(srcgray, dstImage, 3, 3, 0.01, BORDER_DEFAULT);
		//归一化与转换
		normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(normImage, scaledImage);
		Mat binaryImage;
		threshold(scaledImage, binaryImage, 0, 255, THRESH_OTSU | THRESH_BINARY);
	*/

	//2、Shi-Tomasi算法：确定图像强角点
	vector<Point2f> corners;//提供初始角点的坐标位置和精确的坐标的位置
	int maxcorners = 200;
	double qualityLevel = 0.01;  //角点检测可接受的最小特征值
	double minDistance = 10;	//角点之间最小距离
	int blockSize = 3;//计算导数自相关矩阵时指定的领域范围
	double  k = 0.04; //权重系数

	goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
	//Mat():表示感兴趣区域；false:表示不用Harris角点检测

	//输出角点信息
	cout << "角点信息为：" << corners.size() << endl;

	//绘制角点
	RNG rng(12345);
	for (unsigned i = 0; i < corners.size(); i++)
	{
		circle(srcImage, corners[i], 2, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
		cout << "角点坐标：" << corners[i] << endl;
	}

	//3、寻找亚像素角点
	Size winSize = Size(5, 5);  //搜素窗口的一半尺寸
	Size zeroZone = Size(-1, -1);//表示死区的一半尺寸
	//求角点的迭代过程的终止条件，即角点位置的确定
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
	//TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);

	cornerSubPix(srcgray, corners, winSize, zeroZone, criteria);

	//输出角点信息
	cout << "角点信息为：" << corners.size() << endl;

	//绘制角点
	for (unsigned i = 0; i < corners.size(); i++)
	{
		circle(srcImage, corners[i], 2, Scalar(255, 0, 0), -1, 8, 0);
		cout << "角点坐标：" << corners[i] << endl;
	}
#endif // coner

	std::cout << "finish debuging...";
	return 0;
}