#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

//��ȷ��λ�ǵ����ꡣ
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
	//�ṩ��ʼ�ǵ������λ�ú;�ȷ�������λ��
	int maxcorners = 200;
	double qualityLevel = 0.01;
	//�ǵ���ɽ��ܵ���С����ֵ
	double minDistance = 10;	//�ǵ�֮����С����
	int blockSize = 3;//���㵼������ؾ���ʱָ��������Χ
	double  k = 0.04; //Ȩ��ϵ��
	goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);	//Mat():��ʾ����Ȥ����false:��ʾ����Harris�ǵ��� 	//����ǵ���Ϣ
	cout << "�ǵ���ϢΪ��" << corners.size() << endl; 	//���ƽǵ�
	RNG rng(12345);
	for (unsigned i = 0; i < corners.size(); i++)
	{
		circle(srcImage, corners[i], 2, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
		cout << "�ǵ����꣺" << corners[i] << endl;
	}

	return 0;
}
//��������������������������������
//��Ȩ����������ΪCSDN����������.�š���ԭ������
//ԭ�����ӣ�https ://blog.csdn.net/xinyuski/article/details/93472253