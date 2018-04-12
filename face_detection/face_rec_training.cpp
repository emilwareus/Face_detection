#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
using namespace std;


using namespace std;
using namespace cv;

Mat subtractMean(Mat mat);
Mat covMatrix(Mat mat, bool isMeanSubtracted = true);
Mat pca(Mat mat, bool isColumnFeatures = true);

const string searchPattern = "images/*.jpg";

Mat pcaOpencv(Mat mat) {
  cout << "PCA..." << endl;
  PCA pca(mat, Mat(), PCA::DATA_AS_COL, 200);
  Mat eigenvalues = pca.eigenvalues.clone();
  Mat eigenvectors = pca.eigenvectors.clone();
  cout << "EIGENVECTORS: " << endl << eigenvectors.rows << " " << eigenvectors.cols << endl;
  cout << "MEAN: " << endl << pca.mean.rows << "   " << pca.mean.cols << endl;
  Mat a = eigenvectors.reshape(1, 100);
  Mat mean = pca.mean.col(0).reshape(1, 100);
  Mat covariance;
  calcCovarMatrix(mat, covariance, pca.mean, CV_COVAR_NORMAL | CV_COVAR_COLS, CV_32F);
  cout << "COVARIANCE:" << endl;
  for (int i =0; i < 10; i++) {
    cout << covariance.at<float>(i, 0) << "    ";
  }
  cout << endl << "EIGEN: " << endl; 
//  cout << a.rows << "   " << a.cols << endl << mean.rows << "   " << mean.cols << endl;;
  for (int i = 0; i < eigenvalues.rows; i++) {
    cout << eigenvalues.at<float>(i, 0) << endl;
  }
//  imshow("asdf", mean.t() * a / 255 );
//  waitKey(0);
  return eigenvectors;
}

/** Function Headers */
/*
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
  // cout << stacked.rows << "    " << stacked.cols;
  pcaOpencv(stacked);
  Mat eigenvectors = pca(stacked, true);
  return 0;
}
*/


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
  covar = (1.0/mat.cols) * (mat * mat.t());
  cout << covar.rows << covar.cols; 
  return covar;
}

Mat pca(Mat mat, bool isColumnFeatures) {
  cout << "Calculating covariance..." << endl;
  Mat covariance = covMatrix(mat, false, isColumnFeatures);
  for (int i =0; i < 10; i++) {
    cout << covariance.at<float>(i, 0) << "    ";
  }
  Mat eigenvals;
  Mat eigenvecs;
  cout << "Calculating eigen..." << endl;
  eigen(covariance, eigenvals, eigenvecs);
  
  for (int i = 0; i < 5; i++) {
    cout << eigenvals.at<float>(i, 0) << "   " << endl;
  }
  return eigenvecs;
}


/**
Reads a CSV file with "filename" with the eigenvalues.
The CSV needs to be structured:
nrCol
nrRow
X;X;X;X;X;X....
X;X;X;X;X;X....
.
.
.

Returns the read CSV in a 2d vector format in string.
*/
vector< vector<string> > laod_pretrained(const string& filename) {
	string line;
	ifstream file(filename.c_str());
	string delimiter = ";";
	string token, token_row;

	getline(file, line, ',');
	int col = stoi(line.substr(0, line.find(delimiter)));
	line.erase(0, col);
	int row = stoi(line.substr(0, line.find(delimiter)));
	line.erase(0, col + 1);


	vector<vector<string>> output_matrix(row, vector<string>(col));


	int i = 0;
	size_t pos_row = 0;

	while ((pos_row = line.find('\n')) != string::npos) {

		if (pos_row != 0) {
			token_row = line.substr(0, pos_row);
			size_t pos = 0;
			int j = 0;

			while ((pos = token_row.find(delimiter)) != string::npos) {
				token = token_row.substr(0, pos);
				token_row.erase(0, pos + delimiter.length());
				output_matrix[i][j] = token;
				j++;

			}
			token = token_row.substr(0, pos);
			token_row.erase(0, pos + delimiter.length());
			output_matrix[i][j] = token;
			cout << endl;
			i++;
		}
		line.erase(0, pos_row + 1);

	}


	return output_matrix;
}


void laod_pretrained(vector< vector<string> > save_matrix, const string& filename) {

	int rows = save_matrix.size();
	int cols = save_matrix[0].size();

	ofstream myfile;
	myfile.open(filename);
	myfile << cols << "\n";
	myfile << rows << "\n";

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			myfile << save_matrix[i][j] << ";";
		}
		myfile << "\n";
	}
	myfile.close();

}


/*
float euclidean_distance(vector<String> v1, vector<String>  v2) {
	float dist = 0;
	for (int i = 0; i < v1.size(); i++) {
		dist += (v1[i] - v2[i])*(v1[i] - v2[i]);
	}
	return sqrt(dist);

}*/