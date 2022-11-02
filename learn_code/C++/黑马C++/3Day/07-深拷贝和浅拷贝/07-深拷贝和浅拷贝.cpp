#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<string>
using namespace std;

class Person {
public:
	// 构造函数
	Person() {}	//默认构造
	// 有参构造，目的是初始化属性
	Person(const char *name,int age) {
		m_Name = (char*)malloc(strlen(name) + 1);  // 开辟空间
		strcpy(m_Name, name);
		m_Age = age;
	}
	// 拷贝，系统会提供默认简单值拷贝

	// 析构，释放类对象以及堆上的属性（指针，动态分配空间）
	// 浅拷贝析构函数释放内存会出现系统崩溃，
	// 因为浅拷贝只拷贝地址，但是会释放堆区空间两次，需要使用深拷贝
	// 自己构造深拷贝构造解决
	Person(const Person& p) {
		m_Age = p.m_Age;
		m_Name = (char*)malloc(strlen(p.m_Name) + 1);	// 分配地址
		strcpy(m_Name, p.m_Name);	// cpy值
	}
	~Person() {
		cout << "析构函数调用！" << endl;
		// 判断成员属性是否需要释放，指针是否为空
		if (m_Name != NULL) {
			free(m_Name);	// malloc/free是C/C++语言的标准库函数,new/delete、malloc/free必须配对使用。
			m_Name = NULL;	// 使指针为空，防止野指针
				/*
					概念：野指针就是指向的内存地址是未知的(随机的，不正确的，没有明确限制的)。
					说明：指针变量也是变量，是变量就可以任意赋值。但是，任意数值赋值给指针变量没有意义，
							因为这样的指针就成了野指针，此指针指向的区域是未知
							(操作系统不允许操作此指针指向的内存区域)。
					注：野指针不会直接引发错误，操作野指针指向的内存区域才会出问题。
				*/
		}

	}

	// 姓名
	char* m_Name;
	// 年龄
	int m_Age;
};

void test01() {
	Person p1("老王", 10);
	Person p2(p1);

	cout << "拷贝p2.name:" << p2.m_Name << ", p2.age:" << p2.m_Age << endl;

}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}