#include<iostream>

using namespace std;

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

class Person {
public:
	Person(int age) {
		if (age < 0 || age>150) {
			// 输入年龄越界异常抛出
			throw out_of_range("年龄必须要在 0 到 150 之间！！");
		}
		this->m_Age = age;
	}

	int m_Age;
};

void test01() {
	try {
		Person p1(151);
	}
	//catch (out_of_range& e) {
	//	cout << e.what() << endl;
	//}
	catch (exception& e) {		// 直接使用异常多态的父类
		cout << e.what() << endl;	// 系统标准异常都有一个.what()借口输出异常信息
	}
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}