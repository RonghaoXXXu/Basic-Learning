//#include <highgui.hpp>
//#include <opencv2/opencv.hpp>
//#include "opencv2/videoio.hpp"
//#include <stdlib.h>
//
//using namespace std;
//using namespace cv;
//
//const char* keys = "{ video  | | Path to the input video file. Skip this argument to capture frames from a camera.}";
//
//Rect process_frame(Mat& image, int opts);
//
//int main(int argc, char** argv) {
//	//1.ʶ���������Ƶ�е��ض���ɫ����
//	//2.GrabCutǰ�����ض�������ȡ)
//	VideoCapture capture(0);
//
//	namedWindow("input", WINDOW_AUTOSIZE);
//
//	int fps = capture.get(CAP_PROP_FPS);//֡��
//	int width = capture.get(CAP_PROP_FRAME_WIDTH);//��
//	int height = capture.get(CAP_PROP_FRAME_HEIGHT);//��
//	int num_of_frames = capture.get(CAP_PROP_FRAME_COUNT);//��ͼ����
//
//	printf("frame width: %d, frame height: %d, FPS : %d \n", width, height, fps);
//
//	Mat frame;
//	int index = 0;
//	while (capture.read(frame) && capture.isOpened()) {
//		capture >> frame;
//		imshow("input", frame);
//		char c = waitKey(50);
//		Rect rect = process_frame(frame, index);
//
//		Mat bgd, fgd, result;
//
//		grabCut(frame, result, rect, bgd, fgd, 1, GC_INIT_WITH_RECT);
//		compare(result, GC_PR_FGD, result, CMP_EQ);
//
//		Mat foreground(frame.size(), CV_8UC3, cv::Scalar(255, 255, 255));
//		frame.copyTo(foreground, result); // �����Ʊ�������
//
//		imshow("result2", foreground);
//
//		if (c == 27) {
//			break;
//		}
//	}
//	// Parse command line arguments
//	//CommandLineParser parser(argc, argv, keys);
//
//	//// Create a VideoCapture object and open the input file
//	//VideoCapture cap(0);
//	//if (parser.has("video")) {
//	//	cap.open(parser.get<String>("video"));
//	//}
//	//else
//	//	cap.open(0);
//
//	//// Check if camera opened successfully
//	//if (!cap.isOpened()) {
//	//	cout << "Error opening video stream or file" << endl;
//	//	return -1;
//	//}
//
//	//Mat background;
//	//for (int i = 0; i < 60; i++)
//	//{
//	//	cap >> background;
//	//}
//	////ͼ��ķ�ת,0����x����ת��������������y����ת�����⸺������x��y��ͬʱ��ת��
//	//flip(background, background, 1);
//
//	//while (1)
//	//{
//	//	Mat frame;
//	//	// Capture frame-by-frame
//	//	cap >> frame;
//
//	//	// If the frame is empty, break immediately
//	//	if (frame.empty())
//	//		break;
//
//	//	Mat hsv;
//	//	flip(frame, frame, 1);
//	//	cvtColor(frame, hsv, COLOR_BGR2HSV);
//
//	//	Mat mask1, mask2;
//	//	inRange(hsv, Scalar(0, 0, 0), Scalar(180, 255, 146), mask1);
//	//	inRange(hsv, Scalar(181, 255, 147), Scalar(255, 255, 255), mask2);
//
//	//	mask1 = mask1 + mask2;
//
//	//	Mat kernel = Mat::ones(3, 3, CV_32F);
//	//	morphologyEx(mask1, mask1, cv::MORPH_OPEN, kernel);//��̬ѧ������������
//	//	morphologyEx(mask1, mask1, cv::MORPH_DILATE, kernel);//��ʴ����
//
//	//	//������mask1, mask2��Ϊ�����㣬��mask1�����к�ɫ��λ�ã�mask2����û�к�ɫĻ����λ��
//	//	bitwise_not(mask1, mask2);//bitwise_not��ͼ�񣨻Ҷ�ͼ����ɫͼ����ɣ�ÿ������ֵ���ж����ơ��ǡ�����,mask��Ϊ���
//
//	//	Mat res1, res2, final_output;
//	//	bitwise_and(frame, frame, res1, mask2);//�����㣬res1��Ϊ�����û�к�ɫĻ����λ����������
//	//	bitwise_and(background, background, res2, mask1);//�к�ɫĻ����λ�ã�ʹ�ñ��������
//	//	addWeighted(res1, 1, res2, 1, 0, final_output);
//
//	//	imshow("Magic !!!", final_output);
//	//	// Display the resulting frame
//	//	//imshow( "Frame", frame );
//
//	//	// Press  ESC on keyboard to exit
//	//	char c = (char)waitKey(25);
//	//	if (c == 27)
//	//		break;
//	//	// Also relese all the mat created in the code to avoid memory leakage.
//	//	frame.release(), hsv.release(), mask1.release(), mask2.release(), res1.release(), res2.release(), final_output.release();
//	//}
//
//	//// When everything done, release the video capture object
//	//cap.release();
//
//	//// Closes all the frames
//	//destroyAllWindows();
//
//	waitKey(0);
//	return 0;
//}
//
//Rect process_frame(Mat& image, int opts) {
//	Mat hsv, mask;
//	Rect rect;
//	cvtColor(image, hsv, COLOR_BGR2HSV);
//
//	Mat se = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));//��������
//	inRange(hsv, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);
//	//inRangeʵ�ֶ�ֵ��
//	morphologyEx(mask, mask, MORPH_OPEN, se);//������
//	morphologyEx(mask, mask, MORPH_DILATE, se);
//
//	vector<vector<Point>> contours;
//	vector<Vec4i> hierarchy;
//	findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//	for (size_t t = 0; t < contours.size(); t++) {
//		//�����λ���
//
//		rect = boundingRect(contours[t]);
//		rectangle(image, rect, Scalar(0, 0, 0), 2, 4, 0);
//		imshow("result1", image);
//
//		//RotatedRect rect = minAreaRect(contours[t]);
//
//		////��ȡ��ת���ε��ĸ�����
//		//Point2f* vertices = new Point2f[4];
//		//rect.points(vertices);
//
//		////�����߻���
//		//for (int j = 0; j < 4; j++)
//		//{
//		//	line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
//		//}
//
//		//��������
//
//		/*vector<cv::Point2f> contour_;
//
//		for (int i = 0; i < 4; i++)
//		{
//			contour_.push_back(vertices[i]);
//		}
//		drawContours(image, contour_, 0, Scalar(0, 0, 255), 1);*/
//		/*if (area > max) {
//			max = area;
//			index = t;
//		}*/
//	}//Ѱ���������
//	return rect;
//}