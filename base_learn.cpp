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
{//���ٻ���ͼ
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
	//ֻ�� src.cols&rows;
	//��ͨ�� <uchar> ��ͨ�� <Vec3b>--<Vec3f> double����
	//��ͨ��----cvtColor---��ͨ��
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
			//�Ҷ�ͼ�Ĵ���ʽ
			gary.at<uchar>(row, col) = max(pixel[0], max(pixel[1], pixel[2]));
			//gary.at<uchar>(row, col) = min(pixel[0], min(pixel[1], pixel[2]));
			dst.at<Vec3b>(row, col)[0] = saturate_cast(255 - pixel[0]);
			dst.at<Vec3b>(row, col)[1] = 255 - pixel[1];
			dst.at<Vec3b>(row, col)[2] = 255 - pixel[2];
		}
	}
	//bitwise_not(src, dst);//ȡ��
	//bitwise_or(src, dst, dst);
	//bitwise_and(src, dst, dst);
	//bitwise_xor(src, dst, dst);

	namedWindow("oppose", 0);
	imshow("oppose", gary);
	waitKey(0);
#endif // pixel_operation
#if image_blending;
	//alpha��addWeighted�������ͬ���ͣ�ͬ�ߴ��ͼƬ
	//dst(I) = saturate(src1(I) * alpha + src2(I)* (1-alpha)+ gamma);
	//add ��multiply��subtract����ֱ�Ӽ���

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
	/*ͼ��任���Կ������� :
	���ر任 - �����
	������� - ����
	����ͼ�����ȺͶԱȶ��������ر任�����
	g(i, j) = a*f(i, j) + b������a > 0(0<a<1,�ԱȶȺ����Ƚ��ͣ���֮���෴), b���������*/
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
	/*�����ߡ����Ρ�԰����Բ�Ȼ���������״
		����line(LINE_ _4\LINE_ _8\LINE_ _AA)
		����Բellipse
		������rectangle
		��Բcircle
		���fillPoly*/
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
	/*ģ��ԭ��
	Smooth / Blur��ͼ��������򵥺ͳ��õĲ���֮һ
	ʹ�øò�����ԭ��֮һ��Ϊ�˸�ͼ��Ԥ����ʱ���������
	��ʹ��Smooth/Blur�����䱳������ѧ�ľ������
	��ͨ����Щ������Ӽ��㶼�����Բ���, �����ֽ������˲���mask��kernel��
	���磺1,��һ�������˲�(��ֵ�˲���Ȩ�����) 2����˹�˲�����̬�ֲ���
		  3,��ֵ��Max����Min���˲�(���ٽ���������������)	4,��˹˫���˲�(���ش���뾶��-1�������)
		  ksize,xsymax,ysymax	 ������>0
	���⣺
	���ֵģ���޷��˷���Ե������Ϣ��ʧȱ�ݡ�ԭ���Ǿ�ֵ�˲��ǻ���ƽ��Ȩ��
	���˹ģ�����ֿ˷� �˸�ȱ��,�����޷���ȫ���⣬��Ϊû�п�������ֵ�Ĳ�ͬ
	���˹˫��ģ��-�Ǳ�Ե�������˲�����,�����˱�Ե��Ϣ��ʧ,������ͼ����������*/
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
	/*ͼ����̬ѧ���� - ������״��һϵ��ͼ��������ĺϼ�,
	��Ҫ�ǻ��ڼ����ۻ����ϵ���̬ѧ��ѧ
	��̬ѧ���ĸ��������� : ���g�����͡�������
	�����븯ʴ��ͼ��������õ���̬ѧ�����ֶ�*/
	/*���ͣ�
	�������������,������ͼ��A�ͽṹԪ��BI�ṹԪ
	��B��A�����ƶ�,����B����������Ϊê��,����B��
	����A���������ֵ�����滻ê�������,����B��Ϊ
	�ṹ�������������״*/
	/*��ʴ��
	���ƣ���Сֵ�滻*/
	//���ƽṹԪ��B
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(10, 100), Point(-1, -1));
	//dilate(src, src, kernel);
	erode(src, src, kernel);
	cv::namedWindow("t1", 0);
	cv::imshow("t1", src);
	cv::waitKey(0);

#endif // eroding_dilating
#if morphology
	/*������:�ȸ�ʴ�����͡�ȥ��С���塣
	- int OPT- CV_MOP_OPEN/ CV_ MOP_CLOSE/ CV_ MOP_ GRADIENT
	CV_ MOP_ _TOPHAT/ CV_ MOP_ BLACKHAT ��̬ѧ��������
	MORPH_ERODE    = 0, //��ʴ
	MORPH_DILATE   = 1, //����
	MORPH_OPEN     = 2, //����������ɫ��С�հ�
	MORPH_CLOSE    = 3, //�ղ���,��ɫ��С���ա�
	MORPH_GRADIENT = 4, //�ݶȲ���.����-��ʴ
	MORPH_TOPHAT   = 5, //��ñ����,ԭͼ��-������
	MORPH_BLACKHAT = 6, //��ñ����
	MORPH_HITMISS  = 7  */
	/*Ӧ�ã�����ˮƽ�ߺʹ�ֱ��
	adaptiveThreshold(
	Mat src, //����ĻҶ�ͼ��
	Matdest, //��ֵͼ��
	double maxValue,//��ֵͼ�����ֵ
	int adaptiveMethod//����Ӧ������ֻ������֮һһ
	// ADAPTIVE THRESH MEAN C��ADAPTIVE THRESH_ GAUSSIAN C
	int thresholdType//��������
	int blockSize,//���С
	doubleC //����C����������,0,����*/
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
	//�������������ֳ���

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
		//1��Harris�ǵ���
		cornerHarris(srcgray, dstImage, 3, 3, 0.01, BORDER_DEFAULT);
		//��һ����ת��
		normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(normImage, scaledImage);
		Mat binaryImage;
		threshold(scaledImage, binaryImage, 0, 255, THRESH_OTSU | THRESH_BINARY);
	*/

	//2��Shi-Tomasi�㷨��ȷ��ͼ��ǿ�ǵ�
	vector<Point2f> corners;//�ṩ��ʼ�ǵ������λ�ú;�ȷ�������λ��
	int maxcorners = 200;
	double qualityLevel = 0.01;  //�ǵ���ɽ��ܵ���С����ֵ
	double minDistance = 10;	//�ǵ�֮����С����
	int blockSize = 3;//���㵼������ؾ���ʱָ��������Χ
	double  k = 0.04; //Ȩ��ϵ��

	goodFeaturesToTrack(srcgray, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);
	//Mat():��ʾ����Ȥ����false:��ʾ����Harris�ǵ���

	//����ǵ���Ϣ
	cout << "�ǵ���ϢΪ��" << corners.size() << endl;

	//���ƽǵ�
	RNG rng(12345);
	for (unsigned i = 0; i < corners.size(); i++)
	{
		circle(srcImage, corners[i], 2, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
		cout << "�ǵ����꣺" << corners[i] << endl;
	}

	//3��Ѱ�������ؽǵ�
	Size winSize = Size(5, 5);  //���ش��ڵ�һ��ߴ�
	Size zeroZone = Size(-1, -1);//��ʾ������һ��ߴ�
	//��ǵ�ĵ������̵���ֹ���������ǵ�λ�õ�ȷ��
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
	//TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);

	cornerSubPix(srcgray, corners, winSize, zeroZone, criteria);

	//����ǵ���Ϣ
	cout << "�ǵ���ϢΪ��" << corners.size() << endl;

	//���ƽǵ�
	for (unsigned i = 0; i < corners.size(); i++)
	{
		circle(srcImage, corners[i], 2, Scalar(255, 0, 0), -1, 8, 0);
		cout << "�ǵ����꣺" << corners[i] << endl;
	}
#endif // coner

	std::cout << "finish debuging...";
	return 0;
}