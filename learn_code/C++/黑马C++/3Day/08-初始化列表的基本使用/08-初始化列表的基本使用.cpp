#include<iostream>

using namespace std;

class Person {
public:
	//Person(){}	// 默认构造函数
	// 有参构造函数
	// 用来初始化数据
	// 方法1：
	//Person(int a, int b, int c){
	//	m_A = a;
	//	m_B = b;
	//	m_C = c;
	//}
	// 方法2：利用初始化列表初始化数据，可以传参
	// 方式： 构造函数之后 + 属性1（参数值），属性2（参数值）。。。。。。
	Person(int a, int b, int c) :m_A(a), m_B(b), m_C(c) {}
	
	// 初始化列表来构造函数，并赋予默认参数
	// 固定参数，不能传参
	Person() :m_A(10), m_B(20), m_C(30) {}


	int m_A;
	int m_B;
	int m_C;
};
void test01() {
	Person p1(10, 20, 30);

	cout << "p1的m_A: " << p1.m_A << ", m_B: " << p1.m_B << ", m_C: " << p1.m_C << endl;

	Person p2;
	cout <<"初始化列表默认构造函数，赋值：\n" << "p2的m_A: " << p2.m_A << ", m_B: " << p2.m_B << ", m_C: " << p2.m_C << endl;

}


int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;

}