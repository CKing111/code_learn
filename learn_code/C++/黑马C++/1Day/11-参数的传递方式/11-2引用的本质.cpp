#include<iostream>

using namespace std;

// 引用的本质是在C++内部实现的指针常量
// Type & ref = val;-----> Type* const ref = &val;
// 构建一个引用，默认情况下会自动生成一个指向引用值的指针常量

void test01(int& ref){	// 引用传递
	ref = 100;			// ref是引用，默认转换为*ref = 100，指针赋值
}

int main() {
	int a = 10;	
	int& aRef = a;		// 自动转化为int* const aRef = &a;这也说明引用必须要有初始化			
	aRef = 20;			// 自动转化为*aRef = 20；指针赋值
	cout << "a: " << a << endl;
	cout << "aRef: " << aRef << endl;
	test01(a);
	cout << "test01(a): " << a << endl;
	system("pause");
	return EXIT_SUCCESS;
}