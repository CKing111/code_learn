#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <algorithm>
#include <functional>
/*
�����㷨 ��������Ԫ��
@param beg ��ʼ������
@param end ����������
@param _callback  �����ص����ߺ�������
@return ��������
*/

void myPrint(int val)
{
	cout << val << endl;
}

class MyPrint
{
public:
	void operator()(int val)
	{
		cout << val << endl;
		m_count++;
	}

	int m_count = 0;
};

void test01()
{
	vector<int>v;
	for (int i = 0; i < 10;i++)
	{
		v.push_back(i);
	}
	
	//for_each(v.begin(), v.end(), myPrint);

	for_each(v.begin(), v.end(), MyPrint());
}
// for_each�з���ֵ
void test02()
{

	vector<int>v;
	for (int i = 0; i < 10; i++)
	{
		v.push_back(i);
	}
	MyPrint mp = for_each(v.begin(), v.end(), MyPrint());

	cout <<"count = " <<  mp.m_count << endl;

}

// for_each���԰󶨲��� �������
class MyPrint3:public binary_function<int ,int , void >
{
public:
	void operator()(int val ,int start) const
	{
		cout << val + start << endl;
		
	}
};

void test03()
{
	vector<int>v;
	for (int i = 0; i < 10; i++)
	{
		v.push_back(i);
	}

	for_each(v.begin(), v.end(), bind2nd(MyPrint3(), 1000));

}



/*
transform�㷨 ��ָ����������Ԫ�ذ��˵���һ������
ע�� : transform �����Ŀ�����������ڴ棬������Ҫ������ǰ������ڴ�
@param beg1 Դ������ʼ������
@param end1 Դ��������������
@param beg2 Ŀ��������ʼ������
@param _cakkback �ص��������ߺ�������
@return ����Ŀ������������
*/
class MyTransform
{
public:
	int operator()(int val)
	{
		return val ;
	}
};

void test04()
{
	vector<int>v1;
	for (int i = 0; i < 10;i++)
	{
		v1.push_back(i);
	}

	vector<int>vTarget;
	//����ָ�� vTarget��С
	vTarget.resize(v1.size());

	transform(v1.begin(), v1.end(), vTarget.begin(), MyTransform());

	for_each(vTarget.begin(), vTarget.end(), [](int val){cout << val << endl; });
}



int main(){

	//test01();
	//test02();
	//test03();
	test04();
	system("pause");
	return EXIT_SUCCESS;
}