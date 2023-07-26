#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;

//1、函数对象 很像函数调用方式 ，因此也称为 仿函数
class MyPrint
{
public:

	void operator()(int num)
	{
		cout << num << endl;
		m_Count++;
	}
	int m_Count = 0;
};

void myPrint(int num)
{
	cout << num << endl;
}

void test01()
{
	MyPrint mp;
	mp(100); //类似函数的调用

	myPrint(100);
}


//2、函数对象 超出普通函数的概念，内部可以拥有自己的状态
void test02()
{
	MyPrint mp;
	mp(100);
	mp(100);
	mp(100);
	mp(100);

	cout << "count = " << mp.m_Count << endl;

}

//3、函数对象可以作为函数的参数传递
void doWork(MyPrint mp , int num)
{
	mp(num);
}
void test03()
{
	doWork(MyPrint(), 1000);
}

int main(){


	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}