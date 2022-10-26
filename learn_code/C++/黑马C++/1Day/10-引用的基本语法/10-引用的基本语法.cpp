#include<iostream>

using namespace std;

// 引用就是起别名，给存储空间起一个可以调用的别名
// 引用可以看作是加了限制的指针，比指针安全
// 用&符号，在变量左侧就是引用，右侧就是取地址（解引用）
// 1.基本语法：Type &别名 = 原名
void test01() {
	int a = 10;
	int& b = a; // 形参b = 实参a,操作b等价于操作a

	b = 20; //改变引用，其原始地址值也发生改变，a也改变

	cout << "a = " << a << endl;
	cout << "&b = " << b << endl;
}
// 2.引用必须初始化
void test02() {
	//int& a;   不可以，未初始化
	int a = 10;
	int& b = a; //引用初始化后就不能修改了,不能变成别人的别名

	int c = 20;
	//b = c;   //这是赋值，是可以改变b的
	//int& b = c;  // 不可以，“b”: 重定义；多次初始化
	cout << "b = " << b << endl;
}
// 3.对数组进行建立引用
void test03() {
	// 初始化数组。
	int arr[10];
	for (int i = 0; i < 10; i++) {
		arr[i] = i;
	}

	// 给数组引用别名
	// 第一种方式
	int(&pArr)[10] = arr;
	// 打印
	for (int i = 0; i < 10; i++) {
		cout << pArr[i] << " " << endl;
	}

	// 第二种方式
	typedef int(ARRAYREF)[10];		//表示声明了一个具有10个元素的int类型数组
									// typedef 为C语言的关键字，作用是为一种数据类型定义一个新名字
	ARRAYREF& pArr2 = arr;
	// 打印
	for (int i = 0; i < 10; i++) {
		cout << pArr2[i] << " " << endl;
	}
}


int main() {
	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;

}