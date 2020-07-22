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
//	//1.识别与跟踪视频中的特定颜色对象
//	//2.GrabCut前景（特定对象提取)
//	VideoCapture capture(0);
//
//	namedWindow("input", WINDOW_AUTOSIZE);
//
//	int fps = capture.get(CAP_PROP_FPS);//帧率
//	int width = capture.get(CAP_PROP_FRAME_WIDTH);//宽
//	int height = capture.get(CAP_PROP_FRAME_HEIGHT);//高
//	int num_of_frames = capture.get(CAP_PROP_FRAME_COUNT);//总图像数
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
//		frame.copyTo(foreground, result); // 不复制背景数据
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
//	////图像的反转,0代表x轴旋转，任意正数代表y轴旋转，任意负数代表x和y轴同时旋转。
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
//	//	morphologyEx(mask1, mask1, cv::MORPH_OPEN, kernel);//形态学操作，开运算
//	//	morphologyEx(mask1, mask1, cv::MORPH_DILATE, kernel);//腐蚀操作
//
//	//	//在这里mask1, mask2互为逆运算，即mask1代表有红色的位置，mask2代表没有红色幕布的位置
//	//	bitwise_not(mask1, mask2);//bitwise_not对图像（灰度图像或彩色图像均可）每个像素值进行二进制“非”操作,mask做为输出
//
//	//	Mat res1, res2, final_output;
//	//	bitwise_and(frame, frame, res1, mask2);//与运算，res1作为输出，没有红色幕布的位置正常保留
//	//	bitwise_and(background, background, res2, mask1);//有红色幕布的位置，使用背景像素填补
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
//	Mat se = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));//自设卷积核
//	inRange(hsv, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);
//	//inRange实现二值化
//	morphologyEx(mask, mask, MORPH_OPEN, se);//开运算
//	morphologyEx(mask, mask, MORPH_DILATE, se);
//
//	vector<vector<Point>> contours;
//	vector<Vec4i> hierarchy;
//	findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//	for (size_t t = 0; t < contours.size(); t++) {
//		//正矩形绘制
//
//		rect = boundingRect(contours[t]);
//		rectangle(image, rect, Scalar(0, 0, 0), 2, 4, 0);
//		imshow("result1", image);
//
//		//RotatedRect rect = minAreaRect(contours[t]);
//
//		////获取旋转矩形的四个顶点
//		//Point2f* vertices = new Point2f[4];
//		//rect.points(vertices);
//
//		////逐条边绘制
//		//for (int j = 0; j < 4; j++)
//		//{
//		//	line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
//		//}
//
//		//轮廓绘制
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
//	}//寻找最大轮廓
//	return rect;
//}