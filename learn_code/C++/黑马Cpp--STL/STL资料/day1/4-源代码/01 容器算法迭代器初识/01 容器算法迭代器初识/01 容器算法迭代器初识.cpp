#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector> //vector容器的头文件
#include <algorithm> //系统标准算法头文件
#include <string>

//普通指针也是属于一种迭代器
void test01()
{
	int arr[5] = { 1, 5, 2, 7, 3 };
	int * p = arr;
	for (int i = 0; i < 5; i++)
	{
		//cout << arr[i] << endl;

		cout << *(p++) << endl;
	}
}

void myPrint(int val)
{
	cout << val << endl;
}

//内置属性类型
void test02()
{
	
	vector<int>v; //声明一个vector的容器 

	//想容器中添加数据
	v.push_back(10);
	v.push_back(20);
	v.push_back(30);
	v.push_back(40);

	//通过迭代器可以 遍历容器
	//每个容器都有自己专属的迭代器
	//vector<int>::iterator itBegin = v.begin(); //起始迭代器

	//vector<int>::iterator itEnd = v.end(); //结束迭代器

	//第一种遍历方式
	//while (itBegin != itEnd)
	//{
	//	cout << *itBegin << endl;
	//	itBegin++;
	//}

	//第二种遍历方式
	//for (vector<int>::iterator it = v.begin(); it != v.end();it++)
	//{
	//	cout << *it << endl;
	//}

	//第三种遍历方式  利用系统提供算法
	for_each(v.begin(), v.end(), myPrint);

}

//自定义数据类型
class Person
{
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}
	string m_Name;
	int m_Age;
};
void test03()
{
	vector<Person> v;

	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);

	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);

	//遍历
	for (vector<Person>::iterator it = v.begin(); it != v.end();it++)
	{
		// *it --- Person类型    it  --- 指针
		cout << "姓名： " << (*it).m_Name << " 年龄： " << it->m_Age << endl;
	}
}

//存放自定义数据类型的指针
void test04()
{
	vector<Person*> v;

	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);

	v.push_back(&p1);
	v.push_back(&p2);
	v.push_back(&p3);
	v.push_back(&p4);

	for (vector<Person*>::iterator it = v.begin(); it != v.end(); it++)
	{
		//*it   ---  Person *
		cout << "姓名： " << (*it)->m_Name << " 年龄： " << (*it)->m_Age << endl;
	}
}

//容器嵌套容器
void test05()
{
	vector< vector<int> > v ;//类似二维数组

	vector<int>v1;
	vector<int>v2;
	vector<int>v3;

	for (int i = 0; i < 10;i++)
	{
		v1.push_back(i);
		v2.push_back(i + 10);
		v3.push_back(i + 100);
	}

	//将小容器 插入到大容器中
	v.push_back(v1);
	v.push_back(v2);
	v.push_back(v3);


	for (vector<vector<int>>::iterator it = v.begin(); it != v.end();it++)
	{
		//*it --- vector<int>
		for (vector<int>::iterator vit = (*it).begin(); vit != (*it).end();vit++)
		{
			//*vit  --- int
			cout << *vit << " ";
		}
		cout << endl;
	}
}

int main(){
	//test01();
	//test02();
	//test03();
	//test04();
	test05();
	system("pause");
	return EXIT_SUCCESS;
}