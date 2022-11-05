#include<iostream>
#include<string>
using namespace std;


class Person {
public:
	Person() { cout << "默认构造函数调用！" << endl; }
	Person(int a) { cout << "有参构造函数调用！" << endl; }
	~Person() { cout << "析构函数调用！" << endl; }

};

void test01() {
	//Person p1;				// 栈区开辟，自动释放空间
	Person* p1 = new Person;	// 堆区开辟空间，不会自动释放空间

	/*
		所有new出来的对象，都会返回相同类型的指针
		malloc 返回的是void*指针，且还需要强制转换类型，声明空间大小
		malloc不会自动调用构造函数，new会自动调用构造函数
		new是运算符，malloc是函数
		使用delete运算符释放堆区空间
		malloc-free，new-delete
	*/
	delete p1;
}

void test02() {
	void * p = new Person;		// 用void* 指针去接受new生成的指针，不会自动释放
	delete p;
}

void test03() {
	// 通过new开辟数组
	Person* pArray = new Person[10];	// 堆区开辟，会自动调用10次默认构造函数
	//delete pArray;
	// 注意：new开辟数组一定会调用默认构造，所以一定要提供默认构造
	// 栈区开辟空间，可以指定有参构造，堆区不可以
	Person pArray2[2] = { Person(1),Person(2) };		// 隐式构造

	// 释放堆区数组
	delete[] pArray;		// 加[]，暗示系统这是一个多目标对象数组，系统会自动寻找调用几次析构函数释放空间
}
int main() {
	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}