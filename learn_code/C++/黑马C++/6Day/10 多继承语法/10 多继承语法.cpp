#include<iostream>

using namespace std;

class Base1 {
public:
	Base1() {
		this->m_A = 10;
	}

	int m_A;
};

class Base2 {
public:
	Base2() {
		this->m_B = 20;
		this->m_A = 20;
	}
	int m_B;
	int m_A;
};

class Son :public Base1, public Base2 {
public:

	int m_C;
	int m_D;
};

/*class Son       size(20):
        +---
 0      | +--- (base class Base1)
 0      | | m_A
        | +---
 4      | +--- (base class Base2)
 4      | | m_B
 8      | | m_A
        | +---
12      | m_C
16      | m_D
        +---*/
void test01() {
	cout << sizeof(Son) << endl;
	Son s;
	cout << "Base1中的m_A = " << s.Base1::m_A << endl;
	cout << "Base2中的m_B = " << s.m_B << endl;

	// 不同父类同名成员函数，需要增加作用域
	cout << "Base1中的m_A = " << s.Base1::m_A << endl;
	cout << "Base2中的m_A = " << s.Base2::m_B << endl;

}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}