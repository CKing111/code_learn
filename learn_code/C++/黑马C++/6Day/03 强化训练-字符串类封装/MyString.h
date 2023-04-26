#pragma once
# define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<string>
using namespace std;

class MyString {
	// 友元，重载<<
	friend ostream& operator<<(ostream& cout, MyString str);
	// 友元，重载>>
	friend istream& operator>>(istream& cin, MyString& str);
public:
	//有参构造
	MyString(const char * str);
	/*在 C++ 中，const 是一个关键字，表示“常量”，用于修饰变量、函数参数、成员函数等等。它有以下作用：
	安全性：使用 const 可以防止程序员在不经意间修改变量的值，从而增加程序的安全性。
				例如，如果你将一个常量声明为 const，那么任何尝试修改该常量的行为都会导致编译错误。
	优化：使用 const 还可以帮助编译器对程序进行优化。如果一个变量被声明为 const，那么编译器就可以安全地将其优化为一个常量，
				从而提高程序的效率。
	可读性：使用 const 可以增加代码的可读性。在函数参数中使用 const，可以告诉读者该函数不会修改该参数；
				在成员函数中使用 const，可以告诉读者该函数不会修改对象的状态。
	总之，使用 const 是一个良好的编程习惯，可以提高程序的安全性、可读性和效率。*/
	// 拷贝构造
	MyString(const MyString& str);

	// 重载[]
	char& operator[](int index);

	// 重载=
	MyString& operator=(char * str);			// str = "aaa"
	MyString& operator=(MyString& str);			// str2 = str

	// 重载+，实现字符串拼接，返回值
	MyString operator+(char * str);
	MyString operator+(MyString& str);

	// 重载==
	bool operator==(char* str);
	bool operator==(MyString& str);

	// 析构函数
	~MyString();
private:
	// 指向堆区的字符串指针
	char* pString;

	// 字符串长度
	int m_Size;
};
