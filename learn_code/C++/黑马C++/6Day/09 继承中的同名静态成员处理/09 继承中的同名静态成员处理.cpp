#include<iostream>

using namespace std;


class Base {
public:
	static int m_A;		// 静态成员函数，类内声明，类外初始化，编译阶段分配内存，共享数据

	static void func(){
		cout << "Base中的静态成员函数func" << endl;
	}
	static void func(int a) {
		cout << "Base中的静态成员函数func" <<a<< endl;
	}
};

int Base::m_A = 10;

class Son :public Base {
public:
	static int m_A;
	static void func() {
		cout << "Son中的静态成员函数func" << endl;
	}
};

int Son::m_A = 20;


void test01() {
	Son s;
	// 通过class对象访问静态成员
	cout << s.m_A << endl;
	cout << "Base中的m_A：" << s.Base::m_A << endl;
	// 通过类名访问，静态成员函数以声明
	cout << "通过类名访问Base中的m_A：" << Son::m_A << endl;
	cout << "通过类名访问Base中的m_A：" << Son::Base::m_A << endl;	// 

	s.func();
	Son::func();
	s.Base::func();
	s.Base::func(10);

	Son::Base::func();
	Son::Base::func(10);
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}
