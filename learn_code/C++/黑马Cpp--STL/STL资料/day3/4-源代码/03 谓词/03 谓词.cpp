#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <algorithm>

//谓词  普通函数 或者仿函数 返回值 是 bool类型，这样的函数或者仿函数称为谓词
//一元谓词
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

	//找到第一个大于20的数字
	vector<int>::iterator pos = find_if(v.begin(), v.end(), GreaterThen20());

	if (pos != v.end())
	{
		cout << "找到大于20的数字为： "<< *pos << endl;
	}
	else
	{
		cout << "未找到" << endl;
	}
}


//二元谓词
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

	//从大到小 排序
	sort(v.begin(), v.end(), MyCompare());

	//[](){} 匿名函数  lambda 
	for_each(v.begin(), v.end(), [](int val){ cout << val << endl; });
}

int main(){

	//test01();
	test02();


	system("pause");
	return EXIT_SUCCESS;
}