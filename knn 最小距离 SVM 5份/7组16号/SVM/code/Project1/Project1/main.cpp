#include <fstream>
#include <sstream>
#include <iostream>
#include<math.h>
#include<map>
#include<opencv2\opencv.hpp> 
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


void read_test_data(float train_data[45][11])
{
	ifstream inf;
	inf.open("test.txt", ifstream::in);
	const int cnt = 11;
	string line;
	float i = 0;
	int j = 0, k1 = 0, k2 = 0;
	
	size_t comma = 0;
	size_t comma2 = 0;
	while (!inf.eof())
	{
		k2 = 0;
		getline(inf, line);
		comma = line.find(',', 0);
		i = atof(line.substr(0, comma).c_str());
		//cout << i << ' '; 
		train_data[k1][k2] = i;
		k2 = k2 + 1;
		while (comma < line.size() && j != cnt - 1)
		{
			comma2 = line.find(',', comma + 1);
			i = atof(line.substr(comma + 1, comma2 - comma - 1).c_str());
			//cout << i << ' '; 
			train_data[k1][k2] = i;
			k2 = k2 + 1;
			++j;
			comma = comma2;
		}
		//cout << endl; 
		k1 = k1 + 1;
		j = 0;
	}
	inf.close();
}

void read_train_data(float train_data[75][11])
{
	ifstream inf;
	inf.open("train.txt", ifstream::in);
	const int cnt = 11;
	string line;
	float i = 0;
	int j = 0, k1 = 0, k2 = 0;

	size_t comma = 0;
	size_t comma2 = 0;
	while (!inf.eof())
	{
		k2 = 0;
		getline(inf, line);
		comma = line.find(',', 0);
		i = atof(line.substr(0, comma).c_str());
		//cout << i << ' '; 
		train_data[k1][k2] = i;
		k2 = k2 + 1;
		while (comma < line.size() && j != cnt - 1)
		{
			comma2 = line.find(',', comma + 1);
			i = atof(line.substr(comma + 1, comma2 - comma - 1).c_str());
			//cout << i << ' '; 
			train_data[k1][k2] = i;
			k2 = k2 + 1;
			++j;
			comma = comma2;
		}
		//cout << endl; 
		k1 = k1 + 1;
		j = 0;
	}
	inf.close();
}

void gui_yi_hua_train(float A[75][11], int M,int N)
{
	int i, j;
	float sum = 0;
	float mean[11];
	float var[11];
	for (j = 1; j < N-1; j++)
	{
		sum = 0;
		for (i = 0; i < M; i++)
		{
			sum = sum + A[i][j];
		}
		mean[j] = sum / M;
		sum = 0;
		for (i = 0; i < M; i++)
		{
			sum = sum + (A[i][j] - mean[j])*(A[i][j] - mean[j]);
		}
		var[j] = sum / M;
	}
	for (j = 1; j < N-1; j++)
	{
		for (i = 0; i < M; i++)
		{
			A[i][j] = (A[i][j] - mean[j]) / var[j];
		}
	}

}

void gui_yi_hua_test(float A[45][11], int M, int N)
{
	int i, j;
	float sum = 0;
	float mean[11];
	float var[11];
	for (j = 1; j < N-1 ; j++)
	{
		sum = 0;
		for (i = 0; i < M; i++)
		{
			sum = sum + A[i][j];
		}
		mean[j] = sum / M;
		sum = 0;
		for (i = 0; i < M; i++)
		{
			sum = sum + (A[i][j] - mean[j])*(A[i][j] - mean[j]);
		}
		var[j] = sum / M;
	}
	for (j = 1; j < N-1; j++)
	{
		for (i = 0; i < M; i++)
		{
			A[i][j] = (A[i][j] - mean[j]) / var[j];
		}
	}

}

int main()
{
	float train[75][11];
	float train_data[75][9];
	float train_label[75];
	float test[45][11];
	float test_data[45][9];
	float test_label[45], acc = 0;
	int k1, k2;


	read_train_data(train);
	read_test_data(test);
	gui_yi_hua_train(train, 75, 11);
	gui_yi_hua_test(test, 45, 11);

	/*for (k1 = 0; k1 < 102; k1++)
	{
		for (k2 = 0; k2 < 14; k2++)
		{
			cout << test[k1][k2] << " " ;
		}
		cout << endl;
	}*/
	
	for (k1 = 0; k1 < 75; k1++)
	{
		train_label[k1] = train[k1][10];
		cout << train_label[k1] << " ";
		for (k2 = 0; k2 < 9; k2++)
		{
			train_data[k1][k2] = train[k1][k2+1];
			cout << train_data[k1][k2] << " ";
		}
		cout << endl;
		
	}

	
	for (k1 = 0; k1 < 45; k1++)
	{
		test_label[k1] = test[k1][10];
		cout << test_label[k1] << " ";
		for (k2 = 0; k2 < 9; k2++)
		{
			test_data[k1][k2] = test[k1][k2 + 1];
			cout << test_data[k1][k2] << " ";
		}	
		cout << endl;
	}

	Mat trainingDataMat(75, 9, CV_32FC1, train_data);
	Mat labelsMat(75, 1, CV_32FC1, train_label);

	// 设置SVM参数
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000000000, 1e-9);

	// 对SVM进行训练
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	SVM.save("svm.xml");
	SVM.load("svm.xml");
	
	Mat result;
	Mat query(45, 9, CV_32FC1, test_data);
	SVM.predict(query, result);

	for (int i = 0; i < 45; i++)
	{
		if (result.at<Point2f>(i, 0).x == test_label[i])
		{
			cout << "第" << i << "个，结果正确！" << endl;
			acc = acc + 1;
		}
		else
		{
			cout << "第" << i << "个，结果错误！" << endl;
			cout << result.at<Point2f>(i, 0).x << " " << test_label[i] << endl;
		}
			
	}
	acc = acc / 45;
	cout << "acc=" << acc << endl;

	
	

	system("pause");
	return 0;
}


