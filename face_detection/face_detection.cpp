#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "face_rec_training.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
cv::Mat detectAndDisplay(Mat frame);

// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved
string filename;

string result_recognition;

float x_location;
float y_location;

// Function main
int main(void)
{
  init();
	VideoCapture capture(0);

	if (!capture.isOpened())  // check if we succeeded
		return -1;

	// Load the cascade
	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading\n");
		return (-1);
	};

	// Read the video stream
	Mat frame;

	for (;;)
	{
		capture >> frame;

		// Apply the classifier to the frame
		if (!frame.empty())
		{
			cv::Mat face = detectAndDisplay(frame);
			//int pixel = face.at<uchar>(10, 20); // example to read pixel(10,20)
			//std::cout << "Pixel coordinate: (10, 20)= " << pixel << std::endl;
			// the detected face is 100x100
			// you may add your NCC code here
			
			//putText(frame, result_recognition, cvPoint(x_location, y_location), FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255, 0, 0), 1, CV_AA);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		int c = waitKey(10);

		if (27 == char(c))
		{
			break;
		}

	}
	return 0;

}


// Function detectAndDisplay
cv::Mat detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	Mat crop;
	Mat res;
	Mat gray;
	string text;
	stringstream sstm;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Set Region of Interest
	cv::Rect roi_b;
	cv::Rect roi_c;

	size_t ic = 0; // ic is index of current element
	int ac = 0; // ac is area of current element

	size_t ib = 0; // ib is index of biggest element
	int ab = 0; // ab is area of biggest element

	for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);

		ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

		if (ac > ab)
		{
			ib = ic;
			roi_b.x = faces[ib].x;
			roi_b.y = faces[ib].y;
			roi_b.width = (faces[ib].width);
			roi_b.height = (faces[ib].height);
		}
		std::cout << "Face location:" << "(" << roi_c.x << ", " << roi_c.y << ")" << std::endl;
		crop = frame(roi_b);
    try {
		resize(crop, res, Size(100, 100), 0, 0, INTER_LINEAR); // This will be needed later while saving images
  } catch( cv::Exception& e ) {
    const char* err_msg = e.what();
      std::cout << "exception caught: " << err_msg << std::endl;
      return frame_gray;
  }
		cvtColor(res, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

		
		// Form a filename
		filename = "";
		stringstream ssfn;
		ssfn << "images_camera/" << filenumber << ".jpg";
		filename = ssfn.str();

		imwrite(filename, gray);
		

		Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);

		/*Print result detection*/
		//x_location = faces[ic].x + faces[ic].height / 5;
		//y_location = faces[ic].y - 15;
		//imshow("original", frame);
		result_recognition = detect_face(gray);
		putText(frame, result_recognition, cvPoint(faces[ic].x + faces[ic].height / 5, faces[ic].y - 15), FONT_HERSHEY_COMPLEX_SMALL, 1.5, cvScalar(255, 0, 0), 1, CV_AA);

	}

	// Show image
	//sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
	//text = sstm.str();

	//putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	//putText(frame, "antoine", cvPointroi_b.width, roi_b.height), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255, 0, 0), 1, CV_AA);
	imshow("original", frame);
	//filename = "";
	//stringstream ssfn;
	//ssfn << "images_camera/" << filenumber << "_1.jpg";
	//filename = ssfn.str();
	//filenumber++;
	/*
	imwrite(filename, frame);

	if (!crop.empty())
	{
	//imshow("detected", crop);
	}
	else
	destroyWindow("detected");
	*/
	return gray;
}
