#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
int main(int argc, const char** argv)
{
  vector<Mat> featureVectors;
  // Read some images/
  for (int i = 0; i < 5; i++) {
    Mat temp;
    temp = imread("images/"+to_string(i)+".jpg", 0);
    imshow("asdf", temp);
    Mat temp_flat = temp.reshape(1, 1);
    featureVectors.push_back(temp_flat);
  }
  // Concatenate images to single matrix
  Mat stacked;
  vconcat(featureVectors, stacked);
  cout << stacked.rows << endl << stacked.cols << endl;
}
