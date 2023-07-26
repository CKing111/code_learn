#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <algorithm>
#include <functional>
#include <string>

//1、函数对象适配器
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
	cout << "请输入起始累加值： " << endl;
	int num;
	cin >> num;
	for_each(v.begin(), v.end(), bind2nd(myPrint(), num));


	//bind1st的参数绑定是相反的
	//for_each(v.begin(), v.end(), bind1st(myPrint(), num));

}

//1、将参数进行绑定  bind2nd
//2、做继承   binary_function<类型1，类型2，返回值类型>
//3、加const


//取反适配器
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
	//取反适配器
	//vector<int>::iterator pos = find_if(v.begin(), v.end(), not1( GreaterThenFive()));

	vector<int>::iterator pos = find_if(v.begin(), v.end(),  not1( bind2nd( greater<int>(),5 )) );

	if (pos != v.end())
	{
		cout << "小于5的数字为：" << *pos << endl;
	}
	else
	{
		cout << "未找到" << endl;
	}

	vector<int>v1;
	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
	}
	sort(v1.begin(), v1.end(), not2( less<int>()));
	for_each(v1.begin(), v1.end(), [](int val){ cout << val << endl; });

}
//取反适配器使用
// 1、一元取反   not1 
// 2、继承  unary_function<类型1 ，返回值类型>
// 3、加const


//函数指针适配器
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
	cout << "请输入起始累加值： " << endl;
	int num;
	cin >> num;
	//函数指针适配器 ，将函数指针  适配成  函数对象
	// ptr_fun
	for_each(v.begin(), v.end(), bind2nd( ptr_fun( myPrint3) ,num)  );
}


//成员函数适配器
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
		cout << "成员函数---- 姓名：　" << m_Name << " 年龄： " << m_Age << endl;
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
//	cout << "姓名：　" << p.m_Name << " 年龄： " << p.m_Age << endl;
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