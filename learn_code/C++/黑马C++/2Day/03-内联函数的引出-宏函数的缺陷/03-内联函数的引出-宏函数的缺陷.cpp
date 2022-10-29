#include<iostream>

using namespace std;

// 宏函数缺陷
// 1. 定义加法
#define MyAdd(x,y) x+y
#define MyAdd2(x,y) (x+y)
#define MAX 1024
// 问题1：定义未严格出现运算错误
void test01() {
	int ret = MyAdd(10, 20);
	cout << "define: ret = " << ret << endl;
	int ret2 = MyAdd(10, 20) * 20;  // 预期结果：（10+20）*20=600
									// 错误结果： 10+20*20 = 410
	cout << "define: ret2 = " << ret2 << endl;
	int ret3 = MyAdd2(10, 20) * 20;  // 预期结果：（10+20）*20=600
	cout << "define: ret3 = " << ret3 << endl;
}

// 问题2：宏元素会重复操作
#define MyCompare(a,b) ((a)<(b)) ? (a):(b)
void test02() {
	int a = 10;
	int b = 20;

	int ret = MyCompare(a, b); // 预期是输出10
	cout << "define: ret = " << ret << endl;
	// 自增元素
	int ret2 = MyCompare(++a, b);// 预期是11，但返回12
								 // （（++a）<(b)）? (++a:b) 
								 // 元素在比较和输出时连续执行力++运算
	cout << "define: ret2 = " << ret2 << endl;
}

// 问题3：宏函数没有作用域

// 内联函数就是用函数的形式来解决上述宏所出现的问题
inline int myadd(int a, int b) { return a + b; }
inline int mycompare(int a, int b) { return a < b ? a : b; }
void test03() {
	int a = 10;
	int b = 20;

	int ret = myadd(a, b);
	int ret2 = mycompare(a, b);
	int ret3 = mycompare(++a, b);
	cout << "myadd, ret = " << ret << endl;
	cout << "mycompare(a,b),ret2 = " << ret2 << endl;
	cout << "mycompare(++a,b),ret3 = " << ret3 << endl;
}

// 1.内联函数注意事项
//		内联函数声明和实现都需要加关键字inline
//		类内部的成员函数，默认变为内联函数
inline void func() {};

int main() {
	//test01();
	//test02();
	test03();
	cout << MAX << endl;
	system("pause");
	return EXIT_SUCCESS;
}