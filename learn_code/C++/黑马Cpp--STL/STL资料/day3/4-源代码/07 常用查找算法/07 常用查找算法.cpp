#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
/*
find算法 查找元素
@param beg 容器开始迭代器
@param end 容器结束迭代器
@param value 查找的元素
@return 返回查找元素的位置
*/

void test01()
{
	vector<int>v1;
	for (int i = 0; i < 10;i++)
	{
		v1.push_back(i);
	}

	//查找有没有5这个元素
	vector<int>::iterator it = find(v1.begin(), v1.end(), 5);
	if (it != v1.end())
	{
		cout << "找到了元素： " << *it << endl;
	}
	else
	{
		cout << "未找到" << endl;
	}

}

class Person
{
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}

	bool operator==(const Person & p)
	{
		return this->m_Name == p.m_Name && this->m_Age == p.m_Age;
	}

	string m_Name;
	int m_Age;
};
void test02()
{
	vector<Person>v;
	Person p1("aaa", 10);
	Person p2("bbb", 40);
	Person p3("ccc", 20);
	Person p4("ddd", 30);

	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);

	vector<Person>::iterator it = find(v.begin(), v.end(), p3);
	if (it != v.end())
	{
		cout << "找到了元素---姓名：  " << (*it).m_Name  << " 年龄： "<< it->m_Age << endl;
	}
	else
	{
		cout << "未找到" << endl;
	}
}

class MyComparePerson :public binary_function<Person*,Person*,bool>
{
public:
	bool operator()(Person * p1 ,Person * p2) const
	{
		return (p1->m_Name == p2->m_Name && p1->m_Age == p2->m_Age);
	}
};

void test03()
{
	vector<Person*>v;
	Person p1("aaa", 10);
	Person p2("bbb", 40);
	Person p3("ccc", 20);
	Person p4("ddd", 30);

	v.push_back(&p1);
	v.push_back(&p2);
	v.push_back(&p3);
	v.push_back(&p4);

	Person * p = new Person("bbb", 40);

	vector<Person*>::iterator it= find_if(v.begin(), v.end(), bind2nd( MyComparePerson() ,p ));
	if (it != v.end())
	{
		cout << "找到了数据 姓名： " << (*it)->m_Name << " 年龄： " << (*it)->m_Age << endl;
	}
	else
	{
		cout << "未找到" << endl;
	}


}


/*
adjacent_find算法 查找相邻重复元素
@param beg 容器开始迭代器
@param end 容器结束迭代器
@param  _callback 回调函数或者谓词(返回bool类型的函数对象)
@return 返回相邻元素的第一个位置的迭代器
*/
void test04()
{
	vector<int>v1;
	v1.push_back(1);
	v1.push_back(4);
	v1.push_back(2);
	v1.push_back(3);
	v1.push_back(4);
	v1.push_back(6);
	v1.push_back(6);
	v1.push_back(17);

	vector<int>::iterator it = adjacent_find(v1.begin(), v1.end());

	if (it != v1.end())
	{
		cout << "相邻的重复元素为：" << *it << endl;
	}
	else
	{
		cout << "未找到" << endl;
	}


}



/*
binary_search算法 二分查找法
注意: 在无序序列中不可用
@param beg 容器开始迭代器
@param end 容器结束迭代器
@param value 查找的元素
@return bool 查找返回true 否则false
*/

void test05()
{
	vector<int>v;
	for (int i = 0; i < 10;i++)
	{
		v.push_back(i);
	}
	//v.push_back(3);

	bool ret = binary_search(v.begin(), v.end(), 9);

	if (ret)
	{
		cout << "找到了" << endl;
	}
	else
	{
		cout << "未找到" << endl;
	}
}


/*
count算法 统计元素出现次数
@param beg 容器开始迭代器
@param end 容器结束迭代器
@param  value回调函数或者谓词(返回bool类型的函数对象)
@return int返回元素个数

count_if算法 统计元素出现次数
@param beg 容器开始迭代器
@param end 容器结束迭代器
@param  callback 回调函数或者谓词(返回bool类型的函数对象)
@return int返回元素个数
*/
class MyCompare6
{
public:
	bool operator()(int val)
	{
		return val > 4;
	}
};
void test06()
{
	vector<int>v;
	for (int i = 0; i < 10; i++)
	{
		v.push_back(i);
	}
	v.push_back(4);
	v.push_back(4);
	v.push_back(4);
	v.push_back(4);

	int num = count(v.begin(), v.end(), 4);
	cout << "4的个数为： " << num << endl;


	//按条件进行统计
	num = count_if(v.begin(), v.end(), MyCompare6());
	cout << "大于4的个数为： " << num << endl;
}



int main(){
	//test01();
	//test02();
	//test03();
	//test04();
	//test05();
	test06();
	system("pause");
	return EXIT_SUCCESS;
}