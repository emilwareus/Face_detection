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
using namespace cv;

#define EIGEN_FACE_COUNT 10

Mat subtractMean(Mat mat, bool isColumnFeatures = true);
Mat covMatrix(Mat mat, bool isMeanSubtracted = true);
Mat pca(Mat mat, bool isColumnFeatures = true);

const string searchPattern = "images/*.jpg";

Mat pcaOpencv(Mat mat) {
  PCA pca(mat, Mat(), PCA::DATA_AS_COL, 200);
  Mat eigenvalues = pca.eigenvalues.clone();
  Mat eigenvectors = pca.eigenvectors.clone();
  Mat a = eigenvectors.reshape(1, 100);
  cout << a.at<float>(0,0);
  Mat mean = pca.mean.col(0).reshape(1, 100);
  Mat covariance;
  cout << endl << "EIGEN: " << endl; 
  for (int i = 0; i < 5; i++) {
    cout << eigenvalues.at<float>(i, 0) << endl;
  }
  Mat meanSubtracted = subtractMean(mat, true);
  
  vector<Mat> eigenfaces;
  for (int i = 0; i < EIGEN_FACE_COUNT; i++) {
    // reshape eigenvector to 100x100
    Mat face = eigenvectors.row(i).t();
    eigenfaces.push_back(face);
  }
  Mat principleEigenfaces;
  hconcat(eigenfaces, principleEigenfaces);
  cout << meanSubtracted.rows << "  " << meanSubtracted.cols << endl;
  cout << principleEigenfaces.rows << "  " << principleEigenfaces.cols << endl;
  Mat datasetReduced = principleEigenfaces.t() * meanSubtracted; 
  return datasetReduced;
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
       Mat resized;
       resize(im, resized, Size(100,100), 0, 0);
       // Flatten image to column vector
       Mat flattened = resized.reshape(1,1).t();
       images.push_back(flattened);
  }
  
  // Concatenate row vectors
  Mat stacked;
  hconcat(images, stacked);
  // cout << stacked.rows << "    " << stacked.cols;
  // Mat testEigen = pcaOpencv(stacked);
  // Get the transformed dataset
  Mat transformedDataset = pca(stacked, true);
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
  covar = (1.0/(mat.cols)) * (mat * mat.t());
  cout << covar.rows << covar.cols; 
  return covar;
}

Mat pca(Mat mat, bool isColumnFeatures) {
  Mat meanSubtracted;
  meanSubtracted = subtractMean(mat, isColumnFeatures);
  cout << "Calculating covariance..." << endl;
  Mat covariance = covMatrix(meanSubtracted, true , isColumnFeatures);
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
  vector<Mat> eigenfaces;
  for (int i = 0; i < EIGEN_FACE_COUNT; i++) {
    // reshape eigenvector to 100x100
    Mat face = eigenvecs.row(i).t();
    eigenfaces.push_back(face);
  }
  Mat principleEigenfaces;
  hconcat(eigenfaces, principleEigenfaces);
  Mat datasetReduced = principleEigenfaces.t() * meanSubtracted; 
  return datasetReduced;
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


	vector<vector<string> > output_matrix(row, vector<string>(col));


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


void save_pretrained(vector< vector<string> > save_matrix, const string& filename) {

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