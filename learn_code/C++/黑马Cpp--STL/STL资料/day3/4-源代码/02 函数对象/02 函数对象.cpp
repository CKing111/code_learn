#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;

//1���������� ���������÷�ʽ �����Ҳ��Ϊ �º���
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
	mp(100); //���ƺ����ĵ���

	myPrint(100);
}


//2���������� ������ͨ�����ĸ���ڲ�����ӵ���Լ���״̬
void test02()
{
	MyPrint mp;
	mp(100);
	mp(100);
	mp(100);
	mp(100);

	cout << "count = " << mp.m_Count << endl;

}

//3���������������Ϊ�����Ĳ�������
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