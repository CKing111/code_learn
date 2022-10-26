#include<iostream>
using namespace std;

// 交换数据
// 1. 值传递，不改变原始值
void mySwap(int a , int b) {
	int tmp = a;
	a = b;
	b = tmp;

	cout << "mySwap::a = " << a << endl;
	cout << "mySwap::b = " << b << endl;
}
void test01() {
	int a = 10;
	int b = 20;
	mySwap(a, b);  //值传递

	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
}

// 2.地址传递
void mySwap2(int* a, int* b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
	cout << "mySwap2::a = " << *a << endl;
	cout << "mySwap2::b = " << *b << endl;
}
void test02() {
	int a = 10;
	int b = 20;
	mySwap2(&a, &b);	// 地址传递
	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
}


// 3.引用传递，类似地址传递
// 操纵别名的方式操纵原始值，形参=实参
// 引用就是指针常量操作
void mySwap3(int& a, int& b) {		//&a = a
	int tmp = a;
	a = b;
	b = tmp;
}
void test03() {
	int a = 10;
	int b = 20;
	mySwap3(a, b);	//引用传递
	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
}


// 引用的注意事项
// 1.引用必须引一块合法的内存空间；
// 2.不要返回局部变量的引用；
int doWork() {
	int a = 10;
	return a;
}
void test04() {
	// int &a = 10;

	//int& ret = doWork();	// 不可以，局部变量引用
	//cout << "ret = " << ret << endl;
	//cout << "ret = " << ret << endl;
	//cout << "ret = " << ret << endl;
	//cout << "ret = " << ret << endl;
}

// 3.函数返回值是引用，那么这个函数可以作为等式左值
int& doWork2() {
	static int a = 10; // 当前文件的全局变量
	return a;
}
void test05() {
	// int &a = 10;
	int&ret = doWork2();	// 可以，函数返回值引用
	cout << "ret = " << ret << endl;
	cout << "ret = " << ret << endl;
	cout << "ret = " << ret << endl;
	cout << "ret = " << ret << endl;
	doWork2() = 1000; //等价于a=1000
	cout << "a = " << doWork2() << endl;

}
//int main() {
	//test01();
	//test02();
	//test03();
	//test04();
	//test05();
	//system("pause");
	//return EXIT_SUCCESS;
//}