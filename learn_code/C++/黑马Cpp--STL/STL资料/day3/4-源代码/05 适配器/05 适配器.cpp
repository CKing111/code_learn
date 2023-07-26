#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <algorithm>
#include <functional>
#include <string>

//1����������������
class myPrint : public binary_function<int,int,void>
{
public:
	void operator()(int val ,int start) const
	{
		cout << "val = " << val << " start = " << start << " sum = " <<  val + start << endl;
	}

};

void test01()
{
	vector<int>v;
	for (int i = 0; i < 10;i++)
	{
		v.push_back(i);
	}
	cout << "��������ʼ�ۼ�ֵ�� " << endl;
	int num;
	cin >> num;
	for_each(v.begin(), v.end(), bind2nd(myPrint(), num));


	//bind1st�Ĳ��������෴��
	//for_each(v.begin(), v.end(), bind1st(myPrint(), num));

}

//1�����������а�  bind2nd
//2�����̳�   binary_function<����1������2������ֵ����>
//3����const


//ȡ��������
class GreaterThenFive :public unary_function<int,bool>
{
public:
	bool operator()(int val) const
	{
		return val > 5;
	}
};
void test02()
{
	vector<int>v;
	for (int i = 0; i < 10;i++)
	{
		v.push_back(i);
	}
	//ȡ��������
	//vector<int>::iterator pos = find_if(v.begin(), v.end(), not1( GreaterThenFive()));

	vector<int>::iterator pos = find_if(v.begin(), v.end(),  not1( bind2nd( greater<int>(),5 )) );

	if (pos != v.end())
	{
		cout << "С��5������Ϊ��" << *pos << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}

	vector<int>v1;
	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
	}
	sort(v1.begin(), v1.end(), not2( less<int>()));
	for_each(v1.begin(), v1.end(), [](int val){ cout << val << endl; });

}
//ȡ��������ʹ��
// 1��һԪȡ��   not1 
// 2���̳�  unary_function<����1 ������ֵ����>
// 3����const


//����ָ��������
void myPrint3(int val ,int start) 
{
	cout << val + start << endl;
}
void test03()
{
	vector<int>v;
	for (int i = 0; i < 10; i++)
	{
		v.push_back(i);
	}
	cout << "��������ʼ�ۼ�ֵ�� " << endl;
	int num;
	cin >> num;
	//����ָ�������� ��������ָ��  �����  ��������
	// ptr_fun
	for_each(v.begin(), v.end(), bind2nd( ptr_fun( myPrint3) ,num)  );
}


//��Ա����������
class Person
{
public:
	Person(string name,int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}

	void showPerson()
	{
		cout << "��Ա����---- ��������" << m_Name << " ���䣺 " << m_Age << endl;
	}

	void plusAge()
	{
		m_Age++;
	}

	string m_Name;
	int m_Age;
};

//void printPerson( Person &p)
//{
//	cout << "��������" << p.m_Name << " ���䣺 " << p.m_Age << endl;
//}

void test04()
{
	vector<Person>v;
	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);
	Person p5("eee", 50);
	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);
	v.push_back(p5);

	//for_each(v.begin(), v.end(), printPerson);

	for_each(v.begin(), v.end(),  mem_fun_ref( &Person::showPerson));
	for_each(v.begin(), v.end(), mem_fun_ref(&Person::plusAge));
	for_each(v.begin(), v.end(), mem_fun_ref(&Person::showPerson));
}

int main(){

	//test01();
	test02();
	//test03();
	//test04();

	system("pause");
	return EXIT_SUCCESS;
}