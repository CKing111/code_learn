#define _CRT_SECURE_ND_WARNINGS
#include<iostream>
#include"game1.h"
#include"game2.h"
using namespace std;


// 如果两个头文件具有相同的函数，则无法识别会报错
// 可以将头文件中的函数定义一个命名空间，使用::调用

// namespace命名空间主要用来解决命名冲突的问题
// 1.命名空间下可以放函数、变量、结构体、类
// 2.命名空间必须要定义在全局作用域下
// 3.命名空间可以嵌套命名空间
// 4.命名空间是开放的，可以随时向原空间添加内容，同名空间会合并
// 5.无命名空间、匿名空间
// 6.命名空间可以其别名

namespace A {
	void Func();
	int m_A = 20;
	struct Person {

	};
	class Animal {};
	namespace B {
		int m_A = 10;
	}
}
namespace A {
	int m_B = 1999;
}
void test01() {
	LOL::goAtk();
	WZ::goAtk();
}
void test02() {
	cout << "作用域B下的m_A为：" << A::B::m_A<<endl;
}
void test03() {
	cout << "作用域A下的A::m_A:" << A::m_A << ", m_B:" << A::m_B << endl;
}

//匿名空间，相当于声明了两个静态变量
namespace {
	int m_C = 100;  // static int m_C, static int m_D
	int m_D = 200;
}

// 别名
namespace veryLongName {
	int m_E = 0;
}
void test04() {
	// 别名
	namespace veryShortName = veryLongName;
	cout << veryLongName::m_E << endl;
	cout << veryShortName::m_E << endl;
}
int main() {
	test01();
	test02();
	test03();
	test04();
	system("pause");
	return EXIT_SUCCESS;
}