#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

void read_test_data(float train_data[33][5])
{
	ifstream inf;
	inf.open("test.txt", ifstream::in);
	const int cnt = 5;
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

void read_train_data(float train_data[48][5])
{
	ifstream inf;
	inf.open("train.txt", ifstream::in);
	const int cnt = 5;
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

void gui_yi_hua_train(float A[48][5], int M,int N)
{
	int i, j;
	float sum = 0;
	float mean[4];
	float var[4];
	for (j = 0; j < N-1; j++)
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
	for (j = 0; j < N - 1; j++)
	{
		for (i = 0; i < M; i++)
		{
			A[i][j] = (A[i][j] - mean[j]) / var[j];
		}
	}

}

void gui_yi_hua_test(float A[33][5], int M, int N)
{
	int i, j;
	float sum = 0;
	float mean[4];
	float var[4];
	for (j = 0; j < N - 1; j++)
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
	for (j = 0; j < N - 1; j++)
	{
		for (i = 0; i < M; i++)
		{
			A[i][j] = (A[i][j] - mean[j]) / var[j];
		}
	}

}

void classify(float test_data[33][5], float mean[3][4])
{
	int i = 0, j = 0;
	float dis1 = 0, dis2 = 0, dis3 = 0;
	float res;
	float acc = 0;

	for (i = 0; i < 33; i++)
	{
		dis1 = 0, dis2 = 0, dis3 = 0;
		for (j = 0; j < 4; j++)
		{
			dis1 = dis1 + (test_data[i][j] - mean[0][j]) *(test_data[i][j] - mean[0][j]);
			dis2 = dis2 + (test_data[i][j] - mean[1][j]) *(test_data[i][j] - mean[1][j]);
			dis3 = dis3 + (test_data[i][j] - mean[2][j]) *(test_data[i][j] - mean[2][j]);			
		}
		if (dis1 < dis2 && dis1 < dis3)
			res = 1;
		else if (dis2 < dis3 && dis2 < dis1)
			res = 2;
		else
			res = 3;
		if (res == test_data[i][4])
		{
			cout << "第" << i << "个，测试正确！";
			acc = acc + 1;
		}

		else
		{
			cout << "第" << i << "个，测试错误！";
			cout << "	" << dis1 << "	" << dis2 << "	" << dis3 << endl;
			cout << "预测结果：" << res << " 真实结果：" << test_data[i][4];
		}

		cout << endl;
	}
	acc = acc / 33;
	cout << endl << "准确率：" << acc;

}

int main()
{
	float train_data[48][5];
	float test_data[33][5];
	int k1, k2;
	float Mean[3][4];

	read_train_data(train_data);
	read_test_data(test_data);
	
	

	gui_yi_hua_train(train_data, 48, 5);
	gui_yi_hua_test(test_data, 33, 5);

	
	for (k1 = 0; k1 < 3; k1++)
	{
		for (k2 = 0; k2 < 4; k2++)
		{
			Mean[k1][k2] = 0;
			
		}
	}
	
	for (k1 = 0; k1 < 48; k1++)
	{
		for (k2 = 0; k2 < 5; k2++)
		{
			Mean[int(train_data[k1][4])-1][k2] = Mean[int(train_data[k1][4])-1][k2] + train_data[k1][k2];
		}	
	}
	
	for (k1 = 0; k1 < 3; k1++)
	{
		for (k2 = 0; k2 < 4; k2++)
		{
			Mean[k1][k2] = Mean[k1][k2]/48;
		}
	}

	

	classify(test_data, Mean);




	system("pause");
	return 0;
}


