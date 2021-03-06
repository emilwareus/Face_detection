#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "face_rec_training.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>

using namespace std;
using namespace cv;

// How many eigen faces to use
#define EIGEN_FACE_COUNT 200

//Headers
Mat subtractMean(Mat mat, bool isColumnFeatures = false);
Mat covMatrix(Mat mat);
Mat pca(Mat mat, bool isColumnFeatures = true);
int train_pca(const string& filename);
void save_pretrained(Mat *save_matrix, vector<string> * labels, const string& filename);
void laod_pretrained(const string& filename, Mat *values, vector<string> * labels);
int euclidean_distance(Mat eigen_faces, Mat  input_face);
Mat get_eigen_face(Mat input_face, Mat eigenspace);
void load_matrix_from_csv(const string& filename, Mat *eigenspace);
void save_mean(Mat mean, const string& filename);
vector<string> split(const string &s, char delim);

const string searchPattern = "train_images/*.jpg";
const string filename = "eigen_faces.csv";
Mat saved_eigen_faces;
Mat saved_eigen_faces_centroids;
Mat eigenspace;
Mat mean_face;
vector<string> labels, labels_centroid;

// Determine if we should train or perform recognition
void init() {
	while (true) {
		cout << "Press 1 to train and 2 to use the existing training data set? ";
		int x;
		cin >> x;
		cout << endl;
		if (x == 1) {
			train_pca("eigen_faces.csv");
			load_matrix_from_csv("eigenspace.csv", &eigenspace);
			laod_pretrained("eigen_faces_centroid.csv", &saved_eigen_faces, &labels);
			load_matrix_from_csv("mean.csv", &mean_face);
			break;
		}
		else if (x == 2) {
			cout << "Let's use the existing database" << endl;
			load_matrix_from_csv("eigenspace.csv", &eigenspace);
			laod_pretrained("eigen_faces.csv", &saved_eigen_faces, &labels);
			laod_pretrained("eigen_faces_centroid.csv", &saved_eigen_faces_centroids, &labels_centroid);
			load_matrix_from_csv("mean.csv", &mean_face);
			break;
		}


	}
	cout << "Would you like to use Centroids(1) or Normal(2) training-set?";
	int k;
	cin >> k;
	cout << endl;
	// Using the centroid dataset
	if (k == 1) {
		cout << "Using Centroids" << endl;
		saved_eigen_faces = saved_eigen_faces_centroids;
		labels = labels_centroid;
	}
	else {
		cout << "Using Normal" << endl;
	}
}

// Detect the faces
String detect_face(Mat face) {
	Mat input_face_float;
	face.convertTo(input_face_float, CV_32F);
	Mat resized;
  try {
		resize(input_face_float, resized, Size(100, 100), 0, 0);
	} catch( cv::Exception& e ) {
		// Catch the exception
  	const char* err_msg = e.what();
    std::cout << "exception caught 2: " << err_msg << std::endl;
    return "";
	}
	// Get the eigen face and return the test distance
	Mat test_face = get_eigen_face(resized, eigenspace);
	int test_distance = euclidean_distance(saved_eigen_faces, test_face);
	return labels[test_distance];
}

// Projects face onto eigenspace (Subtracts from mean first)
Mat get_eigen_face(Mat input_face, Mat eigenspace) {
	if (input_face.cols != 1 && input_face.rows != 1) {
		input_face = input_face.reshape(1, 1);
	}
	Mat meanSubtracted = input_face - mean_face.t(); 
	Mat out = meanSubtracted * eigenspace;
	return out;
}


/*
@filename: the name of the csv file where the training-features will be saved.
*/
int train_pca(const string& filename)
{
  // Read images from a folder
  String path(searchPattern); //select only jpg
  vector<String> fn;
  vector<Mat> images;
  vector<string> names;
  cv::glob(path,fn,true); // recurse
  cout << "Loading images with labels :" << endl;
  for (size_t k=0; k<fn.size(); ++k)
  {
      // Read the raw image, convert the matrix to float so we can perform PCA
       Mat raw_im = imread(fn[k], 0);
      // Dealing with getting the label from filename
       if (raw_im.empty()) continue; //only proceed if sucsessful
	   string name;

	   auto pos = fn[k].rfind("\\");
	   auto pos_dot = fn[k].rfind(".") - pos -1;
	   if (pos != std::string::npos) {
		   name = fn[k].substr(pos + 1, pos_dot);
		   cout << name << endl;
	   }
	   else {
		   auto pos = fn[k].rfind("/");
		   if (pos != std::string::npos) {
			   name = fn[k].substr(pos + 1, pos_dot);
			   cout << name << endl;
		   }
		   else {
			   name = fn[k];
			   cout << name << endl;
		   }
	   }

       names.push_back(name);

      // Convert image to 32F matrix, then resize to 100x100 image
       Mat im;
       raw_im.convertTo(im, CV_32F);
       Mat resized;
       resize(im, resized, Size(100, 100), 0, 0);
       // Flatten image to column vector
       Mat flattened = resized.reshape(1,1).t();
       images.push_back(flattened);
  }

  labels = names; 
  // Concatenate row vector (Will have 10000 * K matrix)
  Mat stacked;
  hconcat(images, stacked);
  Mat transformedDataset = (pca(stacked));

  // transformedDataset will be NUM_EIGEN_FACES x Training Examples matrix
  cout << "Training Done! " << endl;

  /*Compute the center of each class*/
  Mat centroid_transformedDataset(Size(200, 20), CV_32F);
  vector<string> centroidNames;
  for (int k = 0; k < 20; k++) {
	  vector<string> tokens;
	  tokens = split(names[k * 5], '_');
	  string name = tokens[0];
	  centroidNames.push_back(name);

	  // Extract the name
	  for (int j = 0; j < 200; j++) {
		  float sum = 0;
		  for (int i = 0; i < 5; i++) {
			  sum = sum + transformedDataset.at<float>(k * 5 + i, j);
		  }
		  centroid_transformedDataset.at<float>(k, j) = sum / 5;
	  }
  }

  saved_eigen_faces = transformedDataset;
  saved_eigen_faces_centroids = centroid_transformedDataset;
  labels_centroid = centroidNames;
  cout << "Computation done!" << endl;
	
  while(true) {
		cout << "Would you like to save ? y/n: ";
		char x;
		cin >> x;
		cout << endl;
		if (x == 'y') {
			// Save the centrood file and also the full eigenface dataset
			save_pretrained(&centroid_transformedDataset, &centroidNames, "eigen_faces_centroid.csv");
			save_pretrained(&saved_eigen_faces, &names, filename);
			save_mean(eigenspace, "eigenspace.csv");
			save_mean(mean_face, "mean.csv");
			cout << "Eigen-faces saved to "<< filename << endl;
			break;
		}
		else if(x == 'n'){
			cout << "Ok.. Goodbye" << endl;
			break;
		}
  }
  cout << "Computation of centroid for each individual ... " << endl;
  return 0;
}


// For a NxM matrix, subtracts the column-wise or row-wise mean from all M columns
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
    // Save the mean for future use
	mean_face = columnMean;
  } else {
    // Not actually used, but for completeness
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

// Calculate the covariance matrix for a matrix
Mat covMatrix(Mat mat) {
  Mat covar;
  covar = (1.0/(mat.cols)) * (mat * mat.t());
  return covar;
}

Mat pca(Mat mat, bool isColumnFeatures) {
  // Adust the row mean to get centred matrix
  Mat meanSubtracted;
  meanSubtracted = subtractMean(mat, isColumnFeatures);
  cout << "Calculating covariance..." << endl;
  // Calculates covariance matrix
  Mat covariance = covMatrix(meanSubtracted);
  cout << "Covariance Done!" << endl;
  Mat eigenvals;
  Mat eigenvecs;
  // Calculates eigenvalues
  cout << "Calculating eigen..." << endl;
  eigen(covariance, eigenvals, eigenvecs);
  cout << "Eigen Done!" << endl;

  // Take the top EIGEN_FACE_COUNT eigenfaces (Already sorted)
  cout << "Let's take the " << EIGEN_FACE_COUNT << " largest eigen-face values " << endl;
  vector<Mat> eigenfaces;
  for (int i = 0; i < EIGEN_FACE_COUNT; i++) {
    Mat face = eigenvecs.col(i);
    eigenfaces.push_back(face);
  }

  // Concatenates the eigenfaces 
  hconcat(eigenfaces, eigenspace);
  // Rduce dataset to the EIGEN_FACE_COUNT dimensions
  Mat datasetReduced = meanSubtracted.t() * eigenspace;
  return datasetReduced;
}

/*
@eigen_faces: The saved faces that are pre-trained
@input_face: A 1D Mat that is the face you want to figure out
returns: the index of the predicted face.
*/
int euclidean_distance(Mat eigen_faces, Mat  input_face) {
	if (input_face.cols != 1 && input_face.rows != 1) {
		input_face = input_face.reshape(1, 1).t();
	}

	//Getting first distance
	Mat temp;
	pow((eigen_faces.row(0) - input_face), 2, temp);
	cv::Scalar temp_dist = cv::sum(temp);
	float dist = float(temp_dist[0]);
	int index = 0;
    for (int i = 1; i < eigen_faces.rows; i++) {
			Mat temp;
			pow((eigen_faces.row(i) - input_face), 2, temp);
			cv::Scalar temp_dist = cv::sum(temp);
			if (float(temp_dist[0]) < dist) {
				dist = float(temp_dist[0]);
				index = i;
			}
    }
	return index;

}

/*************************
 *
 * File I/O Functions
 *
 *************************/
 // Utility for dealing with strings
 vector<string> split(const string &s, char delim) {
 	stringstream ss(s);
 	string item;
 	vector<string> tokens;
 	while (getline(ss, item, delim)) {
 		tokens.push_back(item);
 	}
 	return tokens;
 }
 
// Loads a matrix from a csv file
void load_matrix_from_csv(const string& filename, Mat *space) {
	string line;
	ifstream file(filename.c_str());

	string delimiter = ";";
	string token, token_row;


	getline(file, line, ',');
	int col = stoi(line.substr(0, line.find(delimiter)));
	string test = line.substr(0, line.find("\n"));
	line.erase(0, line.find("\n") + 1);
	int row = stoi(line.substr(0, line.find(delimiter)));
	test = line.substr(0, line.find("\n") + 1);
	line.erase(0, line.find("\n") + 1);


	Mat temp_values = cv::Mat(row, col -1, CV_32F);

	int i = 0;
	size_t pos_row = 0;

	while ((pos_row = line.find('\n')) != string::npos) {

		if (pos_row != 0) {
			token_row = line.substr(0, pos_row);
			size_t pos = 0;
			int j = 0;
			pos = token_row.find(delimiter);
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

	*space = temp_values;

	cout << "Loading eigenspace done!" << endl;

}
/**
@filname is the relative path to the file
@*values is a pointer to the matrix with eigen-faces (&values)
@*labels is the vector with labels of the eigen-faces (&labes)
Values are loaded into theses variables

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
	int col = stoi(line.substr(0, line.find(delimiter)));
	string test = line.substr(0, line.find("\n"));
	line.erase(0, line.find("\n") + 1);
	int row = stoi(line.substr(0, line.find(delimiter)));
	test = line.substr(0, line.find("\n") + 1);
	line.erase(0, line.find("\n") + 1);
	
	Mat temp_values = cv::Mat(row, col -1, CV_32F);
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

void save_mean(Mat mean, const string& filename) {
  ofstream myfile;
  myfile.open(filename);
	int rows = mean.rows;
	int cols = mean.cols;
	myfile << (cols + 1) << "\n";
	myfile << rows << "\n";
  for (int i = 0; i < mean.rows; i++) {
    for (int j =0; j < mean.cols; j++) {
      myfile << mean.at<float>(i, j);
			if (j < mean.cols- 1) {
				myfile << ";";
			}
    }
		myfile << "\n";
  }
  myfile.close();
}

/*
@filname is the relative path to the file
@*save_matrix is a pointer to the matrix with eigen-faces (&values)
@*labels is the vector with labels of the eigen-faces (&labes)
Note that labels and save_matrix needs to be in the same order
*/
void save_pretrained(Mat *save_matrix, vector<string> * labels, const string& filename) {

	int rows = (*save_matrix).rows;
	int cols = (*save_matrix).cols;
	ofstream myfile;
	myfile.open(filename);
	myfile << (cols + 1) << "\n";
	myfile << rows << "\n";


	for (int i = 0; i < rows; i++) {
		myfile << (*labels)[i] << ";";
		for (int j = 0; j < cols; j++) {
			myfile << (*save_matrix).at<float>(i, j);
			if (j < cols - 1) {
				myfile << ";";
			}
		}
		myfile << "\n";
	}
	myfile.close();

}
