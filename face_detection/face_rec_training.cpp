#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

Mat subtractMean(Mat mat);
Mat covMatrix(Mat mat);
Mat pca(Mat mat);

const string searchPattern = "images/*.jpg";

/** Function Headers */
int main(int argc, const char** argv)
{
  vector<Mat> featureVectors;
  String path("images/*.jpg"); //select only jpg
  vector<String> fn;
  vector<Mat> images;
  cv::glob(path,fn,true); // recurse
  cout << fn.size() << endl;
  for (size_t k=0; k<fn.size(); ++k)
  {
       Mat im = imread(fn[k], 0);
       if (im.empty()) continue; //only proceed if sucsessful
       // you probably want to do some preprocessing
       Mat flattened = im.reshape(1,1);
       transpose(flattened, flattened);
       images.push_back(flattened);
  }
  // Read some images/
  // Concatenate images to single matrix
  cout << images.at(0).at<float>(0,0) << endl;
  Mat stacked;
  hconcat(images, stacked);
  cout << stacked.rows << endl << stacked.cols << endl;
  cout << stacked.at<float>(500,0) << endl;
  Mat meanAdjust = subtractMean(stacked);
  cout << meanAdjust.at<float>(500,0) << endl;
  waitKey(0);
  return 0;
}


Mat subtractMean(Mat mat) {
  Mat colMean;
  Mat repeated;
  reduce(mat, colMean, 1, CV_REDUCE_AVG);
  Mat meanSubtracted = repeat(colMean, 1, 5);
  repeated = repeat(colMean,1, mat.cols);
  cout << repeated.rows << "    " << repeated.cols << endl;
  cout << mat.rows << "    " << mat.cols << endl;
  subtract(mat, repeated , meanSubtracted); 
  return meanSubtracted;
}
