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

Mat eigenspace;
Mat mean;

Mat get_eigen_face(Mat input_face, Mat eigenspace, Mat mean);
Mat pcaOpencv(Mat mat);
string colVecToString(Mat mat);
vector<string> split(const string &s, char delim);
int train_pca(const string& filename);
Mat subtractMean(Mat mat, bool isColumnMean);
Mat covMatrix(Mat mat, bool isMeanSubtracted, bool isColumnMean);
Mat pca(Mat mat, bool isColumnFeatures);
void load_matrix_from_csv(const string& filename, Mat *space);
void laod_pretrained(const string& filename, Mat *values, vector<string> * labels);
void save_mean(Mat mean, const string& filename);
void save_eigenspace(vector<Mat> eigenfaces, const string& filename);
void save_pretrained(Mat *save_matrix, vector<string> * labels, const string& filename);
int euclidean_distance(Mat eigen_faces, Mat  input_face);

string detect_face(Mat face);
void init();
