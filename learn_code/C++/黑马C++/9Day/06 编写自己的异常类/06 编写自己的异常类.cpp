#include<iostream>
#include<string>
using namespace std;

// 目的是基于系统标准异常改写自己所需要的异常信息

// 使用系统标准异常需要引用
#include<stdexcept>
/*
<stdexcept>中包含以下的异常类：
	std::logic_error：表示程序逻辑错误的异常类。
	std::domain_error：表示参数超出有效域的异常类。
	std::invalid_argument：表示无效参数的异常类。
	std::length_error：表示长度超过限制的异常类。
	std::out_of_range：表示访问超出范围的异常类。
	std::runtime_error：表示运行时错误的异常类。
	std::overflow_error：表示算术上溢出的异常类。
	std::underflow_error：表示算术下溢出的异常类。
*/

// 自己的异常类
class MyOutOfRange :public exception {		// 集成于标准异常的父类
public:
	// 输入错误信息（string）
	MyOutOfRange(char* errorInfo) {
		// 将char* 转换为string
		// 方法一：
		this->m_ErrorInfo = string(errorInfo);
	}
	// 方法2									
	MyOutOfRange(string errorInfo) {
		// 将char* 转换为string
		// 方法一：
		this->m_ErrorInfo = errorInfo;
	}
	// 重写父类的虚函数：~exception()和what()
	virtual ~MyOutOfRange()
	{
	}
	 virtual char const* what() const		// 第二个const代表常函数，修饰this指针
	{
		// string 转为char*
		 return this->m_ErrorInfo.c_str();	//  string 类的 c_str() 成员函数，将当前字符串转换为 C 风格字符串，
	}

	 string m_ErrorInfo;
};

class Person {
public:
	Person(int age) {
		if (age < 0 || age>150) {
			// 输入年龄越界异常抛出
			//throw out_of_range("年龄必须要在 0 到 150 之间！！");
			//throw MyOutOfRange(string("自定义异常：年龄必须要在 0 到 150 之间！！"));		// 输入string
			throw MyOutOfRange("自定义异常：年龄必须要在 0 到 150 之间！！");				// 输入char
		}
		this->m_Age = age;
	}

	int m_Age;
};


void test01() {
	try {
		Person p1(1511);
	}
	catch (exception& e) {		// 直接使用异常多态的父类
		cout << e.what() << endl;	// 系统标准异常都有一个.what()借口输出异常信息
	}
	//catch (MyOutOfRange& e) {		// 
	//	cout << e.what() << endl;	// 系统标准异常都有一个.what()借口输出异常信息
	//}
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}