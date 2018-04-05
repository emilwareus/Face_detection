#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

Mat subtractMean(Mat mat);
Mat covMatrix(Mat mat, bool isMeanSubtracted = true);
Mat pca(Mat mat, bool isColumnFeatures = true);

const string searchPattern = "images/*.jpg";

Mat pcaOpencv(Mat mat) {
  cout << "PCA...";
  PCA pca(mat, Mat(), PCA::DATA_AS_COL, 200);
  Mat eigenvalues = pca.eigenvalues.clone();
//   cout << eigenvalues.rows << " " << eigenvalues.cols << endl;
  Mat eigenvectors = pca.eigenvectors.clone();
//  cout << eigenvectors.rows << " " << eigenvectors.cols << endl;
//  cout << pca.mean.rows << "   " << pca.mean.cols << endl;
  Mat a = eigenvectors.reshape(1, 100);
  Mat mean = pca.mean.col(0).reshape(1, 100);
  cout << a.rows << "   " << a.cols << endl << mean.rows << "   " << mean.cols << endl;;
  for (int i = 0; i < eigenvalues.rows; i++) {
    cout << eigenvalues.at<float>(i, 0) << "   ";
  }
//  imshow("asdf", mean.t() * a / 255 );
//  waitKey(0);
  return eigenvectors;
}

/** Function Headers */
int main(int argc, const char** argv)
{
  // Read images from a folder
  String path(searchPattern); //select only jpg
  vector<String> fn;
  vector<Mat> images;
  cv::glob(path,fn,true); // recurse
  
  for (size_t k=0; k<fn.size(); ++k)
  {
      // Read the raw image, convert the matrix to float so we can perform PCA
       Mat raw_im = imread(fn[k], 0);
       if (raw_im.empty()) continue; //only proceed if sucsessful
       Mat im;
       raw_im.convertTo(im, CV_32F);
       // Flatten image to column vector
       Mat flattened = im.reshape(1,1).t();
       images.push_back(flattened);
  }
  
  // Concatenate row vectors
  Mat stacked;
  hconcat(images, stacked);
  cout << stacked.rows << "    " << stacked.cols;
  pcaOpencv(stacked);
  Mat eigenvectors = pca(stacked, true);
  //  Mat meanAdjust = subtractMean(stacked);
  //  cout << meanAdjust.rows << "   " << meanAdjust.cols << endl;
  //  Mat covarMatrix = covMatrix(meanAdjust);
  //  cout << covarMatrix.rows << "   " << covarMatrix.cols;
  return 0;
}

// For a NxM matrix, subtracts the columwise mean from all M columns
Mat subtractMean(Mat mat, bool isColumnMean) {
  Mat meanSubtracted;
  if (isColumnMean) {
    Mat columnMean;
    Mat repeated;
    // Get row vector with means for each column
    reduce(mat, columnMean, 1, CV_REDUCE_AVG);
    // Tile the means for subtraction
    repeated = repeat(columnMean, 1, mat.cols);
    subtract(mat, repeated, meanSubtracted); 
  } else {
    Mat rowMean;
    Mat repeated;
    // Get row vector with means for each column
    reduce(mat, rowMean, 0, CV_REDUCE_AVG);
    // Tile the means for subtraction
    repeated  = repeat(rowMean, mat.rows, 1);
    subtract(mat, repeated, meanSubtracted); 
  }
  return meanSubtracted;
}

// Calculate the covariance matrix for a matrix with row feature vectors
Mat covMatrix(Mat mat, bool isMeanSubtracted, bool isColumnMean) {
  Mat covar;
  Mat meanRow;
  if (!isMeanSubtracted) {
    mat = subtractMean(mat, isColumnMean);
  }
  // calcCovarMatrix(mat, covar, meanRow, COVAR_COLS, CV_32F);
  covar = (1/mat.cols) * (mat * mat.t());
  cout << covar.rows << covar.cols; 
  return covar;
}

Mat pca(Mat mat, bool isColumnFeatures) {
  cout << "Calculating covariance..." << endl;
  Mat covariance = covMatrix(mat, false, isColumnFeatures);
  Mat eigenvals;
  Mat eigenvecs;
  cout << "Calculating eigen..." << endl;
  eigen(covariance, eigenvals, eigenvecs);
  
  for (int i = 0; i < eigenvals.rows; i++) {
    cout << eigenvals.at<float>(i, 0) << "   ";
  }
  return eigenvecs;
}

