#include<iostream>

using namespace std;

// C++ 异常处理

// 抛出自定义异常
class MyException {
public:
	void printError() {
		cout << "自定义异常类型捕获" << endl; 
	}
};

// 栈解旋
class Person {
public:
	Person() {
		cout << "构造函数" << endl;
	}
	~Person() {
		cout << "析构函数" << endl;
	}
};

int myDivide(int a, int b) {
	// C 语言处理方法
	if (b == 0) {
		//return -1;

		// C++ 中 异常后抛出，只关注抛出异常类型
		//throw 1;
		//throw 3.14;
		//throw "a";

		// 栈解旋：当发生异常时，从进入 try 块后，到异常被抛出前，这期间在栈上构造的所有对象都会被自动析构。
		//			析构的顺序与构造的顺序相反1 2。这一过程可以保证异常安全性，避免内存泄漏或资源占用2。
		Person p1;
		Person p2;
		cout << "---------" << endl;
		/*
			构造函数
			构造函数
			---------
			析构函数
			析构函数
			自定义异常类型捕获
		*/
		throw MyException();		// 抛出一个MyException匿名对象
	}
	return a / b;
}

void test01() {
	int a = 10;
	int b = 0;
	// C语言处理方式
	myDivide(a, b);

	// C++ 处理异常
	try {		// 尝试某可能异常函数，该函数可以抛出异常
		int ret = myDivide(a, b);
		cout << "ret 的结果为：" << ret << endl;
	}
	catch (int ) {		// 捕获函数抛出的异常，int类型
		cout << "int 类型的异常被捕获" << endl;
	}
	catch (double) {		// 捕获函数抛出的异常，double类型
		// 函数捕获异常后，不想再此处理异常，向上抛出
		throw;
		cout << "double 类型的异常被捕获" << endl;
	}
	catch (MyException e) {
		e.printError();
	}
	catch (...) {		// 捕获函数抛出的异常，其他所有类型
		cout << "其他 类型的异常被捕获" << endl;
	}
}



int main() {
	// main函数处理异常
	try {
		test01();
	}
		catch (MyException e) {
		e.printError();
	}
	catch (...) {		// 如果不捕获，程序会自动调用terminate函数终结程序	
		//std::exception_ptr eptr = std::current_exception();
		//if (eptr) {
		//	try {
		//		std::rethrow_exception(eptr);
		//	}
		//	catch (const std::exception& e) {
		//		cout << "main函数捕获处理，" << typeid(e).name() << " 类型的异常" << endl;
		//		//throw;
		//	} 
		//	//catch (...) {
		//	//	cout << "main函数捕获处理，未知 类型的异常" << endl;
		//	//}
		//}
		cout << "main函数捕获处理，其他 类型的异常" << endl;

	}
	system("pause");
	return EXIT_SUCCESS;
}