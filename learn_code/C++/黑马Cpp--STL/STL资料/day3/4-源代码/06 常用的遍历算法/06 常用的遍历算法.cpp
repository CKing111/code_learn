#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <algorithm>
#include <functional>
/*
遍历算法 遍历容器元素
@param beg 开始迭代器
@param end 结束迭代器
@param _callback  函数回调或者函数对象
@return 函数对象
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
// for_each有返回值
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

// for_each可以绑定参数 进行输出
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
transform算法 将指定容器区间元素搬运到另一容器中
注意 : transform 不会给目标容器分配内存，所以需要我们提前分配好内存
@param beg1 源容器开始迭代器
@param end1 源容器结束迭代器
@param beg2 目标容器开始迭代器
@param _cakkback 回调函数或者函数对象
@return 返回目标容器迭代器
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
	//重新指定 vTarget大小
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