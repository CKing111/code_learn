#include<iostream>

using namespace std;

/*
	1.1 静态成员变量
		1.1.1 编译阶段分配内存
		1.1.2 所有对象共享数据
		1.1.3 通过对象、类名访问
		1.1.4 有权限控制
		1.1.5 类内声明、类外初始化
	1.2 静态成员函数
		1.2.1 可以访问静态成员变量，不可以访问普通成员变量
		1.2.2 普通成员函数都可以访问
		1.2.3 静态成员函数也有权限
		1.2.4 可以通过对象、类名访问 
*/



class Person {
public:
	Person() {}
	int m_A;
	static int m_Age;		// static就是静态成员变量，会共享数据
	// 静态成员变量，类内声明，尽量类外进行初始化

	// 静态成员函数
	// 不可以访问普通成员变量，无法明确属性值来自于哪一个共享函数
	// 正常访问共享的静态成员变量，不需要区分数据来源那个共享函数
	static void func() {
		//m_A = 10;		// 失败，普通成员变量
		m_Age = 10;		// 成功，共享成员变量
		cout << "静态成员函数调用！" << endl;
	}

	// 静态成员变量也是有权限的
private:
	static int m_Other;
	static void func2() {
		cout << "私有空间、静态成员函数func2调用！" << endl;
	}
};

int Person::m_Age = 0;		// 类外初始化实现
int Person::m_Other = 10;	// 私有权限在类外可以初始化

void test01() {
	Person p1;
	p1.m_Age = 10;
	cout << "p1的Age：" << p1.m_Age << endl;

	Person p2;
	p2.m_Age = 20;
	// 1.通过对象访问属性
	cout << "p1的Age：" << p1.m_Age << endl;
	cout << "p2的Age：" << p2.m_Age << endl;

	// 2.通过类名访问属性
	cout << "通过类名访问静态变量m_Age：" << Person::m_Age << endl;
	//cout << "m_Other：" << Person::m_Other << endl;  // 失败，私有权限在类外无法访问

	// 静态成员函数调用
	p1.func();
	p2.func();
	Person::func();

	//Person::func2();		// 失败，私有空间
}


int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}