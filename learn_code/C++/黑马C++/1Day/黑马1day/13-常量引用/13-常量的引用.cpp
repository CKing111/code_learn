#include<iostream>

using namespace std;

// 1.常量的引用
// 不常用，看懂即可
void test01() {
	// int &ref = 10; // 不可以，引用了不合法的内存
	const int& ref = 10; // 可以，加入const后编译器会自动引用临时内存，变为合法内存
		// 等价于： int tmp = 10; const int &ref = tmp;
	cout << "原始ref = " << ref << endl;

	// 只要是合法空间内存的都可以进行赋值和修改数值
	// ref = 20;  // 不可以，const固定了常量
	// 使用指针绕开编辑器进行修改
	int* p = (int*)&ref;
	*p = 1000;
	cout << "ref = " << ref << endl;
}

// 2.修饰形参（常量引用使用场景）
// 只想使用传入的形参而不是修改实参，用const修饰
// 暗示不可更改
void showValue(const int &val) {
	// val += 1000;  // 不可更改
	cout << "value: " << val << endl;

	// 可改，但是不道德
	int* p = (int*)&val;
	*p = 1000;
	cout << "指针修改后的value: " << val << endl;
}
void test02() {
	int a = 10;
	showValue(a);
}
int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}