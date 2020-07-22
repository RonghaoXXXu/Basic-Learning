#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

Mat frame, gray;
Mat pre_frame, pre_gray;

vector <Point2f> features;  //��Žǵ�
vector <Point2f> iniPoints; //��ʼ����������
vector <Point2f> fpts[2]; // fpts[0]  fpts[1]  ���浱ǰ֡��ǰһ֡��������λ��
vector <uchar>   status;  //����ʱ����������ٱ�־λ
vector <float>   errors;  //����ʱ ��������ܺ�

//��������
void detectFeatures(Mat& ingray); //���������
void drawFeature(Mat& inFrame);
void kLTrackFearture();
void drawTrackLine();

int main()
{
	//VideoCapture capture(0); // ʹ������ͷ���в�׽
	VideoCapture capture(0);
	//capture.open("demo8.mp4");
	if (!capture.isOpened())
	{
		printf("cloud not find the file\n");
		return -1;
	}

	//����ÿһ֡�Ĵ���
	while (capture.read(frame))
	{
		flip(frame, frame, 1);    //���õ�������ͷ ��תһ��
		cvtColor(frame, gray, COLOR_BGR2GRAY); //����ǰ֡��ת��Ϊ�Ҷ�ͼ��׼��������������

		//ԭ���ڻҶȻ��Ժ��Ӧ�ý�����������ң�Ȼ����и���
		//���ڿ��ǵ���ǰ�����㣬�������˶����̻��ж�ʧ������������趨��ֵ
		//��������С��40��ʱ�򣬽����¼����֡��������
		if (fpts[0].size() < 40)  //�������˶��Ĺ����У��������Խ��Խ�٣�����������ֵ������������������ʱ�����¼��㵱ǰ֡��������
		{
			detectFeatures(gray); //��������꣬�������������features ֮��
			fpts[0].insert(fpts[0].end(), features.begin(), features.end()); //insert ���뺯�������ڶ��������������뵽��һ����������֮��
			iniPoints.insert(iniPoints.end(), features.begin(), features.end());  //��ʼ���������ݣ�����һ�μ�����˵����������������
		}
		else
		{
			printf("�����������.\n");
		}

		//��һ֡��ǰһ֡ͼ��һ���ǿյģ�Ҫ�ѵ�һ֡�ĻҶ�ͼ�� ���� ��ǰһ֡ͼ��
		if (pre_gray.empty())    //���ǰһ֡ͼ��Ϊ�գ�˵���ǵ�һ֡ͼ��
			gray.copyTo(pre_gray);  //copyTo  ��  clone ���ߵĹ�ϵ

		kLTrackFearture(); //����
		drawFeature(frame);

		//����ǰһ֡����
		gray.copyTo(pre_gray);
		frame.copyTo(pre_frame);

		imshow("input_video", frame); //��ʾÿһ֡ͼ��
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
	double minDistance = 10;  //����������֮��ľ���Ϊ10 �����С��10 ����Ϊ��ͬһ��������
	double blockSize = 3;
	goodFeaturesToTrack(ingray, features, maxCorners, qualitylevel, minDistance, Mat(), blockSize, false, 0.04);  //�ǵ���
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
	//���������
	for (size_t i = 0; i < fpts[1].size(); i++)
	{
		double dist = abs(fpts[0][i].x - fpts[1][i].x) + abs(fpts[0][i].y - fpts[1][i].y);
		if (dist > 2 && status[i])
		{
			iniPoints[k] = iniPoints[i];
			fpts[1][k++] = fpts[1][i];
		}
	}
	//����������
	iniPoints.resize(k);  //resize()  �����Ǹı�������С
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