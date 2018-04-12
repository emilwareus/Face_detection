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

#define EIGEN_FACE_COUNT 200

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

string colVecToString(Mat mat) { 
  stringstream ss;
  for (int i = 0; i < mat.rows; i++) {
    ss << mat.at<float>(i, 0);
    if (i < mat.rows - 1) {
      ss << ";";
    } 
  }
  return ss.str();
}

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

/** Function Headers */
int main(int argc, const char** argv)
{
  // Read images from a folder
  String path(searchPattern); //select only jpg
  vector<String> fn;
  vector<Mat> images;
  vector<string> names;
  cv::glob(path,fn,true); // recurse
  
  for (size_t k=0; k<fn.size(); ++k)
  {
      // Read the raw image, convert the matrix to float so we can perform PCA
       Mat raw_im = imread(fn[k], 0);
      
      // Dealing with getting the label from filename 
       if (raw_im.empty()) continue; //only proceed if sucsessful
       size_t lastindex = fn[k].size()-11;
       size_t lastindexslash = fn[k].find("/")+1;
       string rawname = fn[k].substr(lastindexslash, lastindex);
       names.push_back(rawname);

      // Convert image to 32F matrix, then resize to 100x100 image
       Mat im;
       raw_im.convertTo(im, CV_32F);
       Mat resized;
       resize(im, resized, Size(100,100), 0, 0);
       // Flatten image to column vector
       Mat flattened = resized.reshape(1,1).t();
       images.push_back(flattened);
  }
  
  // Concatenate row vector (Will have 10000 * K matrix)
  Mat stacked;
  hconcat(images, stacked);
  // cout << stacked.rows << "    " << stacked.cols;
  // Mat testEigen = pcaOpencv(stacked);
  // Get the transformed dataset
  Mat transformedDataset = pca(stacked, true);
  // transformedDataset will be NUM_EIGEN_FACES x Training Examples matrix
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
  covar = (1.0/(mat.cols)) * (mat * mat.t());
  cout << covar.rows << covar.cols; 
  return covar;
}

Mat pca(Mat mat, bool isColumnFeatures) {
  // Adust the row mean to get centred matrix
  Mat meanSubtracted;
  meanSubtracted = subtractMean(mat, isColumnFeatures);
  cout << "Calculating covariance..." << endl;
  // Calculates covariance matrix
  Mat covariance = covMatrix(meanSubtracted, true , isColumnFeatures);
  Mat eigenvals;
  Mat eigenvecs;
  // Calculates eigenvalues
  cout << "Calculating eigen..." << endl;
  eigen(covariance, eigenvals, eigenvecs);
  
  // Print some eigenvalues for sanity check 
  for (int i = 0; i < 5; i++) {
    cout << eigenvals.at<float>(i, 0) << "   " << endl;
  }
  
  // Take the top EIGEN_FACE_COUNT eigenfaces (Already sorted)
  vector<Mat> eigenfaces;
  for (int i = 0; i < EIGEN_FACE_COUNT; i++) {
    // reshape eigenvector to 100x100
    Mat face = eigenvecs.row(i).t();
    eigenfaces.push_back(face);
  }
  // Concatenates the eigenfaces 
  Mat principleEigenfaces;
  hconcat(eigenfaces, principleEigenfaces);
  // Rduce dataset to the EIGEN_FACE_COUNT dimensions
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
void laod_pretrained(const string& filename, Mat *values, vector<string> * labels) {
	
	string line;
	ifstream file(filename.c_str());
	
	string delimiter = ";";
	string token, token_row;
	

	getline(file, line, ',');
	cout << line<< endl;

	int col = stoi(line.substr(0, line.find(delimiter)));
	line.erase(0, col +1);
	cout << line << endl;
	
	int row = stoi(line.substr(0, line.find(delimiter)));
	cout << "row, col" << row << ", " << col << endl;
	line.erase(0, col + 1);
	cout << line << endl;
	

	

	Mat temp_values = cv::Mat(row, col -1, CV_32F);
	//labels = new vector<string>(row);
	

	vector<vector<string> > output_matrix(row, vector<string>(col));


	int i = 0;
	size_t pos_row = 0;

	while ((pos_row = line.find('\n')) != string::npos) {

		if (pos_row != 0) {
			token_row = line.substr(0, pos_row);
			size_t pos = 0;
			int j = 0;
			pos = token_row.find(delimiter);
			token = token_row.substr(0, pos);
			token_row.erase(0, pos + delimiter.length());
			labels->push_back(token);
			while ((pos = token_row.find(delimiter)) != string::npos) {
				token = token_row.substr(0, pos);
				token_row.erase(0, pos + delimiter.length());
				temp_values.at<float>(i, j) = strtof((token).c_str(), 0);
				j++;
			}
			token = token_row.substr(0, pos);
			token_row.erase(0, pos + delimiter.length());
			temp_values.at<float>(i,j) = strtof((token).c_str(), 0);
			i++;
		}
		line.erase(0, pos_row + 1);

	}

	*values = temp_values;
	
	cout << "Loading done!" << endl;
}



void save_pretrained(Mat *save_matrix, vector<string> * labels, const string& filename) {

	int rows = (*save_matrix).rows;
	int cols = (*save_matrix).cols;
	ofstream myfile;
	myfile.open(filename);
	myfile << (cols + 1) << "\n";
	myfile << rows << "\n";

	
	for (int i = 0; i < rows; i++) {
		cout << (*labels)[i] << ";";
		myfile << (*labels)[i] << ";";
		for (int j = 0; j < cols; j++) {
			cout << (*save_matrix).at<float>(i, j) << ";";
			myfile << (*save_matrix).at<float>(i, j);
			if (j < cols - 1) {
				myfile << ";";
			}
		}
		cout << endl;
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

}

int main() {
	Mat m;
	vector<string> names;
	
	laod_pretrained("test_csv.csv", &m, &names);


	cout << "Trying data, namesize, m.cols" << names.size() << "   " << m.cols << endl;
	for (int i = 0; i < names.size(); i++) {
		cout << names[i] << "  : ";
		for (int j = 0; j < m.cols; j++) {
			m.at<float>(i, j) = m.at<float>(i, j) + 1;
			cout << m.at<float>(i,j) << ", ";
		}
		cout << " " << endl;
	}
	
	char x;
	cin >> x;
	
	save_pretrained(&m, &names,  "test_csv.csv");
	cin >> x;
	return 0;

}
*/