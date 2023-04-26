#include <iostream>

using namespace std;

class Person {
public:
	void showPerson() {
		cout << this->m_A << endl;
		this->m_A = 100;	// 成功
		//this = NULL  // 错误
		// Person * const this
		// this指针的本质就是一个指针常量，指针的指向是不可以改变的，指针的指向值可以改
	}

	//  常函数
	// 成员函数声明后加const代表常函数，不可修改成员属性
	// 除非增加mutable参数的成员变量
	void showPerson_const() const {		// 常函数
		cout << this->m_A << endl;
		//this->m_A = 100;	// 失败，const Person * const this
		this->m_B = 100;	// 成功
	}

	void showPerson2() {
		cout << "aaa" << endl;
	}
	int m_A;
	mutable int m_B;
};

void test01() {
	Person p1;
	p1.m_A = 10;

	p1.showPerson();
	p1.showPerson_const();
}

// 常对象
// 常对象可以操作mutable成员变量，但不能修改非mutable成员变量。
void test02(){
	const Person p2;	// 常对象
	//p2.m_A = 100;		// 失败
	p2.m_B = 100;		// 成功

	p2.showPerson_const();	// 成功，常对象只能调用常函数
	//p2.showPerson2();		// 失败，常对象是不可以调用正常成员函数的

}
int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}