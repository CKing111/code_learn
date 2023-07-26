#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <algorithm>
#include <iterator>
#include <functional>
/*
set_intersection�㷨 ������set���ϵĽ���
ע��:�������ϱ�������������
@param beg1 ����1��ʼ������
@param end1 ����1����������
@param beg2 ����2��ʼ������
@param end2 ����2����������
@param dest  Ŀ��������ʼ������
@return Ŀ�����������һ��Ԫ�صĵ�������ַ
*/


void test01()
{
	vector<int>v1;
	vector<int>v2;

	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
		v2.push_back(i + 5);
	}

	vector<int>vTarget; //Ŀ������
	vTarget.resize( min(v1.size(),v2.size()) );//����������ȡС�Ŀ���size

	vector<int>::iterator itEnd = set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), vTarget.begin());


	copy(vTarget.begin(), itEnd, ostream_iterator<int>(cout, " "));
	cout << endl;
}

/*
set_union�㷨 ������set���ϵĲ���
ע��:�������ϱ�������������
@param beg1 ����1��ʼ������
@param end1 ����1����������
@param beg2 ����2��ʼ������
@param end2 ����2����������
@param dest  Ŀ��������ʼ������
@return Ŀ�����������һ��Ԫ�صĵ�������ַ
*/
void test02()
{
	vector<int>v1;
	vector<int>v2;

	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
		v2.push_back(i + 5);
	}

	vector<int>vTarget;

	vTarget.resize(v1.size() + v2.size());

	vector<int>::iterator itEnd = set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), vTarget.begin());

	copy(vTarget.begin(), itEnd, ostream_iterator<int>(cout, " "));
	cout << endl;
}

/*
set_difference�㷨 ������set���ϵĲ
ע��:�������ϱ�������������
@param beg1 ����1��ʼ������
@param end1 ����1����������
@param beg2 ����2��ʼ������
@param end2 ����2����������
@param dest  Ŀ��������ʼ������
@return Ŀ�����������һ��Ԫ�صĵ�������ַ
*/


void test03()
{
	vector<int>v1;
	vector<int>v2;

	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
		v2.push_back(i + 5);
	}

	vector<int>vTarget;

	vTarget.resize( max(v1.size(),v2.size()) );

	//v1 �� v2 �
	//vector<int>::iterator itEnd = set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(), vTarget.begin());

	//v2 �� v1 �
	vector<int>::iterator itEnd = set_difference(v2.begin(), v2.end(), v1.begin(), v1.end(), vTarget.begin());
	copy(vTarget.begin(), itEnd, ostream_iterator<int>(cout, " "));
	cout << endl;
}



int main(){


	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}