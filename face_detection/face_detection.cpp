#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <ctime>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame, bool cropFace);
void detectAndSave(Mat frame);
void saveImageCallback(int event, int x, int y, int flags, void* userdata);
/** Global variables */
String face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

/** @function main */
/*
int main(int argc, const char** argv)
{
	int camera = 1;
  // If we want to save images, we should only crop the face for the frame
  bool save = false;
  // Argument parsing
	if (argc > 1) {
    for (int i = 1; i < argc; i ++){ 
      char* p;
      long converted = strtol(argv[i], &p, 10);
      if (strcmp(argv[i],"-save") == 0) {
        save = true; 
      } 
      if ((strlen(argv[i]) == 1 && isdigit(argv[i][0]))) {
		    camera = atoi(argv[i]);
      }
    }
	}
	CvCapture* capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	VideoCapture stream1(camera);   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}

	//unconditional loop
	

	if (stream1.isOpened())
	{
		while (true)
		{
			Mat frame;
			stream1.read(frame);
      namedWindow(window_name, WINDOW_AUTOSIZE);
			//-- 3. Apply the classifier to the frame
			if (!frame.empty())
			{
        setMouseCallback(window_name, saveImageCallback, &frame);
				detectAndDisplay(frame, save);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

			int c = waitKey(10);
			if ((char)c == 'c') { break; }
		}
	}
}
*/
void saveImageCallback(int event, int x, int y, int flags, void* userdata) 
{
    if  ( event == EVENT_LBUTTONDOWN)
    {
      Mat* face = (Mat*) userdata;
      Mat image = *face;
      if (!image.empty()) {
        detectAndSave(image);
      } else {
        cout << "Image is empty!";
      }
     }
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame, bool cropFace = false)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);

		Mat faceROI = frame_gray(faces[i]);
    if (cropFace) {
      imshow(window_name, faceROI);
    } else {
        ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
        std::vector<Rect> eyes;
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for (size_t j = 0; j < eyes.size(); j++)
        {
          Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
          int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
          circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
        }
    }
	}
  if (!cropFace) {
	  imshow(window_name, frame);
  }
}

void detectAndSave(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
  cout << "saving";
	for (size_t i = 0; i < faces.size(); i++)
	{
		Mat faceROI = frame_gray(faces[i]);
    Mat faceCropped;
    resize(faceROI, faceCropped, Size(100,100), 0, 0);
    time_t result = time(nullptr);
    stringstream ss;
    ss << result;
    imwrite("images/"+ss.str()+".jpg", faceCropped);
    cout << "Saving image:" << ss.str()+".jpg";  
	}
}
