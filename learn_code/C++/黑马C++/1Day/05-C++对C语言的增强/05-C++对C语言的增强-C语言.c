#include<stdio.h>
#include<string.h>
#include<stdlib.h>

//using namespace std;

// 全局变量增强
// 
int a;
int a = 30;

// 函数检测增强，参数类型增强，返回函数增强
int getRectS(w, h) {
	//return w * h;
};
// 3.类型检测转换增强
void test02() {
	char* p = malloc(sizeof(64)); //malloc万能指针，返回值为void*

}

// 4.struc增强
struct Person {
	int m_Age;
	//void plusAge();	// C语言struct中不可以增加函数
};
void test03() {
	struct Person p1; // C语言可以不增加struct
	//p1.m_Age = 10;
	//p1.plusAge();
	//std::cout << p1.m_Age << std::endl;
}

// 5.bool类型增强 C语言中没有bool类型
// bool flag;
// 
// 6.三目运算符增强，C语言返回的是值
void test05() {
	int a = 30;
	int b = 40;
	printf("ret = %d \n", a > b ? a : b);
	// a > b ? a : b = 100; //报错，20=100，C语言返回的是值
}

// 7.const 增强
const int m_A = 10; //受到保护不可更改
void test06() {
	//m_A = 100;
	const int m_B = 20; //C语言中，可以通过指针改变值，是伪常量
	//m_B = 100;

	int* p = (int*)&m_B;
	*p = 200;
	printf("*p = %d, m_B = %d", *p, m_B);
}
int main() {
	//test01();
	//test02();
	//test05();
	test06();
	system("pause");
	return EXIT_SUCCESS;
}