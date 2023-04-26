#include<iostream>

using namespace std;


class Base {
public:
	Base() {
		this->m_A = 100;
	}
	void func() {
		cout << "Base中的func。" << endl;
	}

	void func(int a) {
		cout << "Base中的func（int a）:" << a << endl;
	};
	int m_A;
};

// 子类中存在同名成员参数，采用就近原则，先子后父
class Son : public Base {
public:
	Son() {
		this->m_A = 200;
	}
	void func() {
		cout << "Son中的func。" << endl;
	}
	int m_A;
};


void test01() {
	Son m;
	cout << m.m_A << endl;				// 就近原则访问子类的值
	cout << "Base中的m_A：(m.Base::m_A )" << m.Base::m_A << endl;		// 继续访问父类，增加作用域

	m.func();		// 就近
	m.Base::func();		// 父类

	//m.func(10);			// 错误，同名的成员函数，就根据就近原则，子类会屏蔽掉父类中的所有类型
	m.Base::func(10);		// 作用域可以获取父类重载版本
}

/*
class Son       size(8):
		+---
 0      | +--- (base class Base)
 0      | | m_A
		| +---
 4      | m_A
		+---N
*/
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;


}