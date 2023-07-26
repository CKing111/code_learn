#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <algorithm>

//ν��  ��ͨ���� ���߷º��� ����ֵ �� bool���ͣ������ĺ������߷º�����Ϊν��
//һԪν��
class GreaterThen20
{
public:
	bool operator()( int val)
	{
		return val > 20;
	}
};

void test01()
{
	vector<int>v;
	v.push_back(10);
	v.push_back(20);
	v.push_back(30);
	v.push_back(40);

	//�ҵ���һ������20������
	vector<int>::iterator pos = find_if(v.begin(), v.end(), GreaterThen20());

	if (pos != v.end())
	{
		cout << "�ҵ�����20������Ϊ�� "<< *pos << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}
}


//��Ԫν��
class MyCompare
{
public:
	bool operator()(int v1 ,int v2)
	{
		return v1 > v2;
	}

};


void test02()
{
	vector<int>v;
	v.push_back(10);
	v.push_back(30);
	v.push_back(20);
	v.push_back(40);

	//�Ӵ�С ����
	sort(v.begin(), v.end(), MyCompare());

	//[](){} ��������  lambda 
	for_each(v.begin(), v.end(), [](int val){ cout << val << endl; });
}

int main(){

	//test01();
	test02();


	system("pause");
	return EXIT_SUCCESS;
}