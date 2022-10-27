#include<iostream>

using namespace std;

// C语言中没有默认参数和占位参数

// 函数默认参数
// 函数参数注意：
//	1.如果一个参数有默认值，那么从这个位置开始，从左往右都必须有默认参数
//	2.传参，如果有参数就用传入参数，没有就用默认值
void func(int c, int a = 10, int b = 10 ) {
	cout << "a + b + c = " << a + b + c << endl;
}
void test01() {
	func(1);
	func(1,2);
}
// 3.如果函数声明里有默认参数，那函数实现中就不能有默认参数
//		只能一个有默认参数，否则会出现重定义
void myFunc(int a = 10, int b = 10);		// 成功，函数声明，有默认参数
//void myFunc(int a = 10, int b = 10) {};		// 失败，函数实现，重定义默认参数

// 函数的占位参数，只声明数据类型
// 如果有占位参数，函数调用之后必须要提供这个参数，但是用不到
// 用途小，重载 ++符号可能有用
// 占位参数可以有默认值
void func2(int a, int = 1) {

}
void test02() {
	func2(10, 1);
}


int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}