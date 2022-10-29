#include<iostream>

using namespace std;

// 函数重（chong）载
//	1.C++中，函数名可以重复
//	2.必须要在同一个作用域下，函数名相同才会发生重载
//	3.函数传入参数个数或者类型或者顺序不同才可以
//	4.函数返回值不可以做为重载条件、依据
//	5.注意: 函数重载和默认参数一起使用，需要额外注意二义性问题的产生
//	6.引用可以作为重载依据，const引用也可以


// 全局作用域
void func() {
	cout << "无参数的func" << endl;
};			
void func(int a) { 
	cout << "有参数的func(int a)" << endl;
};	
void func(double a) {
	cout << "有参数的func(double a)" << endl;
}
void func(double a, int b) {
	cout << "有参数的func(double a,int b)" << endl;
}
void func(int a, double b) {
	cout << "有参数的func(int a,double b)" << endl;
}
//返回类型不同，失败，出现二义性
//int func(int a, double b) {
	//cout << "有参数的func(int a,double b)" << endl;
//}
// class作用域
class Person {
public:
	void func() {
		cout << "Person类中的无参数的func（）" << endl;
	}
};
void test01() {
	func();
	func(1);
	func(1.1);
	func(1.1, 1);
	func(1, 1.1);
	Person p1;
	p1.func();
}

// 注意: 函数重载和默认参数一起使用，需要额外注意二义性问题的产生
void func2(int a) {};
void func2(int a, int b = 10) {};
// void test02() { func2(1) };		// 两个重载，无法明确来源，二义性

// 引用的重载版本
void func3(int& a) { cout << "引用版本func(int  a)" << endl; };
void func3(const int& a) { cout << "const引用func(const int & a )" << endl; };
void test03() {
	int a = 10;
	func3(a);		// 可以
	func3(10);		// 一般引用不可以不可以，引用空间不合法， 加const后可以引用
}


int main() {
	//test01();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}