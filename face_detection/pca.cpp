using namespace std;
using namespace cv;

float* subtractMean(uint8_t arr[]) {
	float mean;
	mean = getMean(arr);
	float result[sizeof(arr) / sizeof(*arr)];	
	for (int i = 0; i < size; i++) {
		result[i] = arr[i] / mean;
	}	
	return result;
}

float getMean(uint8_t arr[]) {
	size = sizeof(arr) / sizeof(*arr);
	int sum = 0;
	for (int i = 0; i < size; i++) {
		sum += arr[i];	
	}	
	float mean = sum / size;
	return mean;
}
