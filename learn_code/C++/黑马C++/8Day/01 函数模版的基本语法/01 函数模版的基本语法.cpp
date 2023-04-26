#include <iostream>

using namespace std;

// int数值交换
void mySwapInt(int& a, int& b) {
	int tem = a;
	a = b;
	b = tem;
	
}
// double数值交换
void mySwapDouble(double & a, double & b) {
	double tem = a;
	a = b;
	b = tem;
}
// 。。。。。。各种类型


// 利用函数模版实现通用函数功能
template<typename T>    // 表示定义一个函数模板，这个函数可以接受任何类型的参数。
// typename 关键字表示待实例化的类型参数，
// T 是这个类型参数的名称。T 就可以替代任何类型，具体的类型将在调用该模板时由编译器推断或者由程序员手动指定。
void mySwap(T& a, T& b) {		// 采用引用直接修改原始数据，不同类型不能运算，不用引用不同类型可以运算
	T tem = a;
	a = b;
	b = tem;
}

// 一下模版指定了一个空函数，不可使用，无法推到T类型
// 必须指定模版T的类型才可使用
template<typename T>
void mySwap2() {};

// 函数模版使用要求：
// 1. 自动类型推导：必须让编译器推导出一致的T才可以使用模版
//		eg：mySwap（ a, x )  // 失败，x为char，a为int，不能推导出一致T类型
// 2. 显示指定类型
//		eg: mySwap<int>(a,b)	// 显示指定类型参数T为int类型
//		eg: mySwap<double>(c,d)	// 显示指定类型参数T为double类型
void test01() {
	int a = 10;
	int b = 20;

	mySwapInt(a, b);

	cout << "int a = " << a << endl;
	cout << "int b = " << b << endl;

	double c = 10.1;
	double d = 20.1;

	mySwapDouble(c, d);

	cout << "double c = " << c << endl;
	cout << "double d = " << d << endl;

	cout << "-------------使用template<typename T>函数模板后----------：" << endl;
	int e = 10;
	int f = 20;
	double g = 10.1;
	double h = 20.1;
	mySwap(e, f);
	mySwap(g, h);
	cout << "T e = " << e << endl;
	cout << "T f = " << f << endl;
	cout << "T g = " << g << endl;
	cout << "T h = " << h << endl;

	mySwap2<double>();		// 成功，指定T类型
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}