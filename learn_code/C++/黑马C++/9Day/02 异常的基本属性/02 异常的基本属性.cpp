#include<iostream>

using namespace std;

// C++ 异常处理

int myDivide(int a, int b) {
	// C 语言处理方法
	if (b == 0) {
		//return -1;

		// C++ 中 异常后抛出
		throw 1;
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
	catch (int ) {		// 捕获函数抛出的异常
		cout << "int 类型的异常被捕获" << endl;
	}
}



int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}