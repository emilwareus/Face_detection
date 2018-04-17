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

//Headers
Mat subtractMean(Mat mat, bool isColumnFeatures = true);
Mat covMatrix(Mat mat, bool isMeanSubtracted = true);
Mat pca(Mat mat, bool isColumnFeatures = true);
int train_pca(const string& filename);
void save_pretrained(Mat *save_matrix, vector<string> * labels, const string& filename);
void laod_pretrained(const string& filename, Mat *values, vector<string> * labels);
int euclidean_distance(Mat eigen_faces, Mat  input_face);
Mat get_eigen_face(Mat input_face, Mat eigenspace, Mat mean);
void save_eigenspace(vector<Mat> eigenfaces, const string& filename);
void load_matrix_from_csv(const string& filename, Mat *eigenspace);
void save_mean(Mat mean, const string& filename);

const string searchPattern = "train_images/*.jpg";
Mat saved_eigen_faces;
vector<string> labels;

int main(int argc, const char** argv) {
  Mat eigenspace;	
  Mat mean;
	train_pca("eigen_faces.csv");
  load_matrix_from_csv("eigenspace.csv", &eigenspace);
  load_matrix_from_csv("mean.csv", &mean);
	laod_pretrained("eigen_faces.csv", &saved_eigen_faces, &labels);
   
  Mat input_face = imread("train_images/obama_happy.jpg");
  Mat input_face_float; 
  input_face.convertTo(input_face_float, CV_32F);
  Mat resized;

  resize(input_face_float, resized, Size(100,100), 0, 0);
	Mat test_face = get_eigen_face(resized, eigenspace, mean);
	int test_distace = euclidean_distance(saved_eigen_faces, test_face);
	cout << "Predicted face : " << labels[test_distace] << endl;
	char x;
	cin >> x;

}


//TODO: verify this
Mat get_eigen_face(Mat input_face, Mat eigenspace, Mat mean) {
	if (input_face.cols != 1 && input_face.rows != 1) {
		input_face = input_face.reshape(1, 1).t();
	}
  Mat meanSubtracted = input_face - mean; 
  cout << eigenspace.rows << "   " << eigenspace.cols << endl; 
  cout << meanSubtracted.rows << "   " << meanSubtracted.cols << endl; 
  return eigenspace * meanSubtracted;
}

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
       resize(im, resized, Size(100,100), 0, 0);
       // Flatten image to column vector
       Mat flattened = resized.reshape(1,1).t();
       images.push_back(flattened);
  }
  
  // Concatenate row vector (Will have 10000 * K matrix)
  Mat stacked;
  hconcat(images, stacked);
  cout << stacked.rows<< " ASDF " << stacked.cols;
  // cout << stacked.rows << "    " << stacked.cols;
  // Mat testEigen = pcaOpencv(stacked);
  // Get the transformed dataset
  Mat transformedDataset = (pca(stacked, true)).t();

  // transformedDataset will be NUM_EIGEN_FACES x Training Examples matrix

  cout << "Training Done! " << endl;
  
  while(true){
	cout << "Would you like to save ? y/n: ";
	char x;
	cin >> x;
	cout << endl;
	if (x == 'y') {
	  
		save_pretrained(&transformedDataset, &names, filename);
		cout << "Eigen-faces saved to "<< filename << endl;
		break;
	}
	else if(x == 'n'){
		cout << "Ok.. Goodbye" << endl;
		break;
	}
  }
  
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
    save_mean(columnMean, "mean.csv");
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
  cout << "CovRows : " <<covar.rows << ",  CovCols : " << covar.cols << endl;
  return covar;
}

Mat pca(Mat mat, bool isColumnFeatures) {
  // Adust the row mean to get centred matrix
  Mat meanSubtracted;
  meanSubtracted = subtractMean(mat, isColumnFeatures);
  cout << "Calculating covariance..." << endl;
  // Calculates covariance matrix
  Mat covariance = covMatrix(meanSubtracted, true , isColumnFeatures);
  cout << "Covariance Done!" << endl;
  Mat eigenvals;
  Mat eigenvecs;
  // Calculates eigenvalues
  cout << "Calculating eigen..." << endl;
  eigen(covariance, eigenvals, eigenvecs);
  cout << "Eigen Done!" << endl;
  // Print some eigenvalues for sanity check 
  cout << "Some Eigen Values: " << endl;
  for (int i = 0; i < 5; i++) {
    cout << eigenvals.at<float>(i, 0) << "   " << endl;
  }
  
  // Take the top EIGEN_FACE_COUNT eigenfaces (Already sorted)
  cout << "Let's take the " << EIGEN_FACE_COUNT << " largest eigen-face values " << endl;
  vector<Mat> eigenfaces;
  for (int i = 0; i < EIGEN_FACE_COUNT; i++) {
    Mat face = eigenvecs.row(i).t();
    eigenfaces.push_back(face);
  }
  save_eigenspace(eigenfaces, "eigenspace.csv");
  // Concatenates the eigenfaces 
  Mat principleEigenfaces;
  hconcat(eigenfaces, principleEigenfaces);
  
  // Rduce dataset to the EIGEN_FACE_COUNT dimensions
  Mat datasetReduced = principleEigenfaces.t() * meanSubtracted; 
  return datasetReduced;
}


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
void save_eigenspace(vector<Mat> eigenfaces, const string& filename) {
  ofstream myfile;
  myfile.open(filename);
	int rows = eigenfaces.size();
	int cols = eigenfaces[0].rows;
	myfile << (cols + 1) << "\n";
	myfile << rows << "\n";
  for (int i = 0; i < eigenfaces.size(); i++) {
    for (int j =0; j < eigenfaces[i].rows; j++) {
      myfile << eigenfaces[i].at<float>(j, 0); 
			if (j < eigenfaces[i].rows- 1) {
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
		//cout << (*labels)[i] << ";";
		myfile << (*labels)[i] << ";";
		cout << (*labels)[i] << endl;
		for (int j = 0; j < cols; j++) {
			//cout << (*save_matrix).at<float>(i, j) << ";";
			myfile << (*save_matrix).at<float>(i, j);
			if (j < cols - 1) {
				myfile << ";";
			}
		}
		//cout << endl;
		myfile << "\n";
	}
	myfile.close();

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
