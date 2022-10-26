#include<iostream>

using namespace std;

// 1.全局变量增强
//
//int a;
int a = 30;

// 2.函数检测增强，参数类型增强，返回函数增强
// 必须指定形参类型，使用时必须数量对应
int getRectS(int w, int h) {
	return w * h;
};
void test01() {
	cout << getRectS(10, 10)<<endl;
};

// 3.类型检测转换增强
// 不同类型间的转换，必须先增加强制转换
void test02() {
	// char* p = malloc(sizeof(64)); //malloc万能指针，返回值为void*
	char* p = (char*)malloc(sizeof(64));
}

// 4.struc增强
struct Person {
	int m_Age;
	void plusAge() { m_Age++; };	// C++的struct中可以增加函数
};
void test03() {
	Person p1; // C++可以不增加struct
	p1.m_Age = 10;
	p1.plusAge();
	cout << p1.m_Age << endl;
}

// 5.bool类型增强 C语言中没有bool类型
bool flag = true;	// 只有真或假，true代表真（非0），false代表假（0）
void test04() {
	cout << sizeof(bool) << endl;
	cout << flag << endl;
	flag = 100;  // 默认转化问1
	cout << flag << endl;
}

// 6.三目运算符增强，C++返回的是变量
void test05() {
	int a = 30;
	int b = 40;

	cout << "ret = " << (a > b ? a : b) << endl;
	(a > b ? a : b) = 100; //通过，b=100，C语言返回的是变量
	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
}

// 7.const 增强
const int m_A = 10; // 全局常量，受到保护，不会改
void test06() {
	//m_A = 100;
	const int m_B = 20; //C++中是真正的常量，不会发生改变
	//m_B = 100;

	int* p = (int*)&m_B;
	*p = 200;
	/*
		C++中等价于开辟了一个临时内存空间存储变量，上式为：
		int tmp = m_B;
		int *p = (int *)&tmp;
		*p指向临时空间，原始常量m_B没有改变；	
	*/
	cout << "*p = " << *p << ", m_B = " << m_B << endl;
}

int main() {
	//test01();
	//test03();
	//test04();
	//test05();
	test06();
	system("pause");
	return EXIT_SUCCESS;
}