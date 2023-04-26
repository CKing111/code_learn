#include<iostream>

using namespace std;

int myPlus1(int a, int b) {
	return a + b;
}

template<typename T>
T myPlus2(T a, T b) {
	return a + b;
}


void test01() {
	int a = 10;
	int b = 20;
	char c = 'c';
	// 普通函数和模版函数的区别
	cout << "普通函数可以进行隐式类型转换，char->int: a+c = 10+'c' = 10 + 99 = " << myPlus1(a, c) << endl;
	cout << "myPlus2(a,c)失败，自动类型推导方式不可以进行隐式类型转换" << endl;
	cout << myPlus2<int>(a, c) << ", myPlus2<int>(a,c)显示指定类型方式成功。" << endl;
}

// 两者调用规则
// 1. 如果普通函数和函数模版可以同时调用，优先选择普通函数，逻辑简单
// 2. 如果想强制调用函数模版中的内容，可以使用空参数列表
template<class T>
void myPrint(T a, T b) {
	cout << "函数模版1调用" << endl;;
}
template<class T>
void myPrint(T a, T b, T c) {
	cout << "函数模版2调用" << endl;
}
void myPrint(int a, int b) {
	cout << "普通函数调用" << endl;
}

void test02(){
	int a = 0;
	int b = 0;
	int c = 0;
	myPrint(a, b);		// 调用普通函数
	myPrint<>(a, b);	// 强制调用模版
	myPrint(a, b, c);	// 函数模版重载
	}
int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}