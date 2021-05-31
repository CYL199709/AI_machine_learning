#include <fstream>
#include <sstream>
#include <iostream>
#include<math.h>
#include<map>

using namespace std;

void read_test_data(float train_data[102][14])
{
	ifstream inf;
	inf.open("test.txt", ifstream::in);
	const int cnt = 14;
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

void read_train_data(float train_data[45][14])
{
	ifstream inf;
	inf.open("train.txt", ifstream::in);
	const int cnt = 14;
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

void gui_yi_hua_train(float A[45][14], int M,int N)
{
	int i, j;
	float sum = 0;
	float mean[14];
	float var[14];
	for (j = 1; j < N; j++)
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
	for (j = 1; j < N; j++)
	{
		for (i = 0; i < M; i++)
		{
			A[i][j] = (A[i][j] - mean[j]) / var[j];
		}
	}

}

void gui_yi_hua_test(float A[102][14], int M, int N)
{
	int i, j;
	float sum = 0;
	float mean[14];
	float var[14];
	for (j = 1; j < N ; j++)
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
	for (j = 1; j < N; j++)
	{
		for (i = 0; i < M; i++)
		{
			A[i][j] = (A[i][j] - mean[j]) / var[j];
		}
	}

}

int GetMinDistIndex(float Distance[]){
	int Index = -1;
	double DistMin = 9999;
	if (Distance == nullptr){
		cout << "error!" << endl;
		return -9999;
	}
	for (int i = 0; i < 45; i++)
	{
		if (Distance[i]<DistMin&&Distance[i] >= 0) {
			DistMin = Distance[i];
			Index = i;
		}
	}
	Distance[Index] = -1;//找出最小值后,将其置为-1
	return Index;
}

int GetMaxSeq(int LabelMinIdx[],int TrainLabels[45])
{
	map<int, int> LabelAppearTime;//key为Label值，value为出现次数
	map<int, int>::iterator iter;
	for (int i = 0; i < 4; i++){
		iter = LabelAppearTime.find(TrainLabels[LabelMinIdx[i]]);
		if (iter != LabelAppearTime.end()) iter->second++;
		else {
			LabelAppearTime.insert(pair<int, int>(TrainLabels[LabelMinIdx[i]], 1));
		}
	}

	int LabelMaxSeq = -1;
	int times = 0;
	for (iter = LabelAppearTime.begin(); iter != LabelAppearTime.end(); iter++){
		if (iter->second>times) {
			times = iter->second;
			LabelMaxSeq = iter->first;
		}
	}
	return LabelMaxSeq;
}

double GetDistance(float Input[13], float TrainData[13])
{
	if (Input == nullptr)
	{
		cout << "error!" << endl;
		return -9999;
	}

	double Distance = 0;
	for (int i = 0; i < 13; i++)
	{
		//cout << Input[i] << " " << TrainData[i] << endl;
		Distance += (Input[i] - TrainData[i]) * (Input[i] - TrainData[i]);
	}

	return sqrt(Distance);
}
int Classify(float test_data[13], float TrainData[45][13], int TrainLabels[45])
{
	
	/*
	算法的基本思想是，计算Input向量与DataSet每个向量的距离，并用一个数组Distance储存。
	找出Distance中的最小的前k个值，在Labels向量中找出对应下标并记录对应Labels值
	找出记录下的Labels值中，出现频率最高的作为返回值。
	*/
	float Distance[45];
	for (int i = 0; i < 45; i++)
	{
		//cout << i << endl;
		Distance[i] = GetDistance(test_data, TrainData[i]);
	}
	int LabelMinIdx[4];
	for (int i = 0; i < 4; i++)
	{
		LabelMinIdx[i] = GetMinDistIndex(Distance);//返回的是Label下标
		//cout << "i:" << i << "LabelMinIdx" << LabelMinIdx[i] << endl;
	}
	return GetMaxSeq(LabelMinIdx, TrainLabels);
}

double CorrectRate(float TestData[102][13], int TestLabels[102], float TrainData[45][13] , int TrainLabels[45])
{
	// 对于每个TestData，利用Classify获得LabelsPredict，再和TestData的真实Label计算正确率
	double CorrectNum = 0;
	for (int i = 0; i < 102; i++)
	{
		if (Classify(TestData[i], TrainData, TrainLabels) == TestLabels[i])
		{
			CorrectNum++;
			cout << "第" << i << "个，正确！" << endl;
		}
		else
		{
			cout << "第" << i << "个，错误！" << Classify(TestData[i], TrainData, TrainLabels) <<" "<< TestLabels[i] << endl;
		}
	}
	double CorrectRate = CorrectNum / 102;
	cout << "CorrectRate = " << CorrectRate << endl;
	return CorrectRate;
}

int main()
{
	float train[45][14];
	float train_data[45][13];
	int train_label[45];
	float test[102][14];
	float test_data[102][13];
	int test_label[102];
	int k1, k2;
	double	acc;
	float Mean[3][14];
	read_train_data(train);
	read_test_data(test);
	gui_yi_hua_train(train, 45, 14);
	gui_yi_hua_test(test, 102, 14);

	/*for (k1 = 0; k1 < 102; k1++)
	{
		for (k2 = 0; k2 < 14; k2++)
		{
			cout << test[k1][k2] << " " ;
		}
		cout << endl;
	}*/
	
	for (k1 = 0; k1 < 45; k1++)
	{
		train_label[k1] = train[k1][0];
		for (k2 = 0; k2 < 13; k2++)
		{
			train_data[k1][k2] = train[k1][k2+1];
			//cout << train_data[k1][k2] << " ";
		}
		//cout << endl;
		
	}

	//cout << endl<<train_data[44][12] << endl;
	for (k1 = 0; k1 < 102; k1++)
	{
		test_label[k1] = test[k1][0];
		for (k2 = 0; k2 < 13; k2++)
		{
			test_data[k1][k2] = test[k1][k2 + 1];
			//cout << test_data[k1][k2] << " ";
		}	
		//cout << endl;
	}

	acc = CorrectRate(test_data, test_label, train_data, train_label);
	cout << "acc=" << acc;



	system("pause");
	return 0;
}


