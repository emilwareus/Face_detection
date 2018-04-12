#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <ctime>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void generate_detectAndDisplay(Mat frame, bool cropFace);
void generate_detectAndSave(Mat frame, String name);

/** Global variables */
String gen_face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
String gen_eyes_cascade_name = "haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier gen_face_cascade;
CascadeClassifier gen_eyes_cascade;
string gen_window_name = "Capture - Face detection";

const string gen_searchPattern = "images/*.jpg";

/** @function detectAndDisplay */
void generate_detectAndDisplay(Mat frame, bool cropFace = false)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	gen_face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);

		Mat faceROI = frame_gray(faces[i]);
		if (cropFace) {
			imshow(gen_window_name, faceROI);
		}
		else {
			ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			std::vector<Rect> eyes;
			//-- In each face, detect eyes
			gen_eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
				int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
				circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
			}
		}
	}
	if (!cropFace) {
		imshow(gen_window_name, frame);
	}
}


void generate_detectAndSave(Mat frame, String name)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	if(!frame.empty()){
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		//-- Detect faces
		gen_face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		
		for (size_t i = 0; i < faces.size(); i++)
		{
			Mat faceROI = frame_gray(faces[i]);
			Mat faceCropped;
			resize(faceROI, faceCropped, Size(100, 100), 0, 0);
			time_t result = time(nullptr);

			imwrite("train_images/" + name, faceCropped);
			cout << "Saving image:" << name;
		}
	}
	else {
		cout << "Frame empty" << endl;
	}
}


/** @function main */
int main(int argc, const char** argv)
{
  
  // If we want to save images, we should only crop the face for the frame
  bool save = true;
  // Argument parsing
	
	CvCapture* capture;
	String name;

	//-- 1. Load the cascades
	if (!gen_face_cascade.load(gen_face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!gen_eyes_cascade.load(gen_eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };



	// Read images from a folder
	String path(gen_searchPattern); //select only jpg
	vector<String> fn;
	vector<Mat> images;
	cv::glob(path, fn, true); // recurse

	for(size_t k = 0; k<fn.size(); ++k){
		
		Mat frame = imread(fn[k], CV_LOAD_IMAGE_COLOR);
		if (frame.empty()) continue;

		namedWindow(gen_window_name, WINDOW_AUTOSIZE);
		//-- 3. Apply the classifier to the frame
		if (!frame.empty())
		{	

			
			auto pos = fn[k].rfind("\\");
			if (pos != std::string::npos) {
				name = fn[k].substr(pos + 1);
				cout << name << endl;
			}
			else {
				auto pos = fn[k].rfind("/");
				if (pos != std::string::npos) {
					name = fn[k].substr(pos + 1);
					cout << name << endl;
				}
				else {
					name = fn[k];
					cout << name << endl;
				}
			}
			
			cout << frame.size() << endl;
			//imshow(gen_window_name, frame);
			generate_detectAndSave(frame, name);

		}
		else
		{
			printf(" --(!) No captured frame -- Break!"); break;
		}


		
		int c = waitKey(10);
		if ((char)c == 'c') { break; }

	}
	
}
