#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <queue>
#include <string>
/*
3.5.3.1 queue构造函数
queue<T> queT;//queue采用模板类实现，queue对象的默认构造形式：
queue(const queue &que);//拷贝构造函数
3.5.3.2 queue存取、插入和删除操作
push(elem);//往队尾添加元素
pop();//从队头移除第一个元素
back();//返回最后一个元素
front();//返回第一个元素

3.5.3.3 queue赋值操作
queue& operator=(const queue &que);//重载等号操作符
3.5.3.4 queue大小操作
empty();//判断队列是否为空
size();//返回队列的大小
*/

class Person
{
public:
	Person(string name, int age) :m_Name(name), m_Age(age)
	{}

	string m_Name;
	int m_Age;
};

void test01()
{
	queue<Person>Q;

	Person p1("aaa",10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);

	//入队
	Q.push(p1);
	Q.push(p2);
	Q.push(p3);
	Q.push(p4);

	while ( !Q.empty())
	{
		//获取队头元素
		Person pFront =  Q.front();
		cout << "队头元素 姓名： " << pFront.m_Name << " 年龄： " << pFront.m_Age << endl;

		//获取队尾元素
		Person pBack = Q.back();
		cout << "队尾元素 姓名： " << pBack.m_Name << " 年龄： " << pBack.m_Age << endl;

		//出队
		Q.pop();
	}

	cout << "队列的大小为： " << Q.size() << endl;

}

int main(){

	test01();

	system("pause");
	return EXIT_SUCCESS;
}