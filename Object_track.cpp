#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

Mat frame, gray;
Mat pre_frame, pre_gray;

vector <Point2f> features;  //存放角点
vector <Point2f> iniPoints; //初始化特征数据
vector <Point2f> fpts[2]; // fpts[0]  fpts[1]  保存当前帧和前一帧的特征点位置
vector <uchar>   status;  //跟踪时候，特征点跟踪标志位
vector <float>   errors;  //跟踪时 区域误差总和

//函数声明
void detectFeatures(Mat& ingray); //特征点查找
void drawFeature(Mat& inFrame);
void kLTrackFearture();
void drawTrackLine();

int main()
{
	//VideoCapture capture(0); // 使用摄像头进行捕捉
	VideoCapture capture(0);
	//capture.open("demo8.mp4");
	if (!capture.isOpened())
	{
		printf("cloud not find the file\n");
		return -1;
	}

	//进行每一帧的处理
	while (capture.read(frame))
	{
		flip(frame, frame, 1);    //调用电脑摄像头 反转一下
		cvtColor(frame, gray, COLOR_BGR2GRAY); //将当前帧，转化为灰度图像，准备进行特征点检测

		//原本在灰度化以后就应该进行特征点查找，然后进行跟踪
		//由于考虑到当前特征点，在物体运动过程会有丢失的情况，所以设定阈值
		//当特征点小于40的时候，将重新计算该帧的特征点
		if (fpts[0].size() < 40)  //在物体运动的过程中，特征点会越来越少，所以设置阈值，当特征点数量较少时，重新计算当前帧的特征点
		{
			detectFeatures(gray); //特征检测完，将特征点存入了features 之中
			fpts[0].insert(fpts[0].end(), features.begin(), features.end()); //insert 插入函数，将第二、三个参数插入到第一个参数数组之中
			iniPoints.insert(iniPoints.end(), features.begin(), features.end());  //初始化特征数据，将第一次检测完了的特征点存入数组中
		}
		else
		{
			printf("特征点检测完成.\n");
		}

		//第一帧的前一帧图像一定是空的，要把第一帧的灰度图像 复制 给前一帧图像，
		if (pre_gray.empty())    //如果前一帧图像为空，说明是第一帧图像
			gray.copyTo(pre_gray);  //copyTo  和  clone 两者的关系

		kLTrackFearture(); //跟踪
		drawFeature(frame);

		//更新前一帧数据
		gray.copyTo(pre_gray);
		frame.copyTo(pre_frame);

		imshow("input_video", frame); //显示每一帧图像
		char c = waitKey(5);
		if (c == 27)
			break;
	}
	capture.release();
	waitKey(0);
	return 0;
}

void detectFeatures(Mat& ingray)
{
	double maxCorners = 5000;
	double qualitylevel = 0.01;
	double minDistance = 10;  //两个特征点之间的距离为10 ，如果小于10 则认为是同一个特征点
	double blockSize = 3;
	goodFeaturesToTrack(ingray, features, maxCorners, qualitylevel, minDistance, Mat(), blockSize, false, 0.04);  //角点检测
	cout << "detect features:" << features.size() << endl;
}

void drawFeature(Mat& inFrame)
{
	for (size_t i = 0; i < fpts[0].size(); i++)
	{
		circle(inFrame, fpts[0][i], 2, Scalar(0, 0, 255), 2, 8, 0);
	}
}

void kLTrackFearture()
{
	//fpts[0]
	calcOpticalFlowPyrLK(pre_gray, gray, fpts[0], fpts[1], status, errors);
	int k = 0;
	//特征点过滤
	for (size_t i = 0; i < fpts[1].size(); i++)
	{
		double dist = abs(fpts[0][i].x - fpts[1][i].x) + abs(fpts[0][i].y - fpts[1][i].y);
		if (dist > 2 && status[i])
		{
			iniPoints[k] = iniPoints[i];
			fpts[1][k++] = fpts[1][i];
		}
	}
	//保存特征点
	iniPoints.resize(k);  //resize()  函数是改变容器大小
	fpts[1].resize(k);
	drawTrackLine();
	swap(fpts[1], fpts[0]);
}

void drawTrackLine()
{
	for (size_t i = 0; i < fpts[1].size(); i++)
	{
		line(frame, iniPoints[i], fpts[1][i], Scalar(0, 255, 0), 1, 8, 0);
		circle(frame, fpts[1][i], 2, Scalar(0, 0, 255), 2, 8, 0);
	}
}