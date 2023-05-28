#include<iostream>

using namespace std;

class MyException {
public:
	MyException() {
		cout << "MyException的构造函数调用" << endl;
	}
	MyException(const MyException& e) {
		cout << "MyException的拷贝构造函数调用" << endl;
	}
	~MyException() {
		cout << "MyException的析构函数调用" << endl;
	}
};

void doWork() {
	//throw MyException();
	//throw & MyException();	// 捕获MyException *e时采用，自动释放
	throw new MyException();	// 堆区，捕获MyException *e时，不会自动释放

}

void test01() {
	try {
		doWork();
	} 
	//catch (MyException &e) {		// 采用引用的方式避免调用拷贝构造，提高运行效率
	//	cout << "MyException的异常捕获" << endl;
	//}
	catch (MyException* e) {		
		cout << "MyException的异常捕获" << endl;
		delete e;
	}
	// MyException e:会调用拷贝构造
	// MyException &e:引用，不会调用拷贝构造，建议
	// MyException *e:指针,直接抛出匿名对象&MyException()，释放掉不可以再操作对象e
	// MyException *e:指针,直接抛出匿名对象new MyException()，释放掉可以再操作对象e，但要手动释放e，和引用效率相同

}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}