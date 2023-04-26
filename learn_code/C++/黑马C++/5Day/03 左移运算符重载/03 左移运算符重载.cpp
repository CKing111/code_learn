#include <iostream>

using namespace std;

class Person {
	// 友元
	friend ostream& operator<<(ostream& cout, Person& p1);
public:
	Person() {};									// 默认构造
	Person(int a, int b) {
		this->m_A = a;
		this->m_B = b;
	}												// 有参构造

	// 成员函数<<运算法重载
	// 本质： p1.operator<< (cout) : p1<<cout
	// 不符合习惯，因此<<运算符不适用成员函数重载
	//void operator<<(ostream& cout) {
	//	
	//}
private:
	int m_A;
	int m_B;

};

// 全局函数<<运算法重载
ostream& operator<<(ostream& cout,Person& p1) {
	cout << "m_A = " << p1.m_A << ", m_B = " << p1.m_B;
	return cout;
}

void test01() {
	Person p1(10, 10);

	//cout << " p1的m_A = " << p1.m_A << ", p1的m_B = " << p1.m_B << endl;

	// 目的：想通过cout<<直接打印Person的参数
	// 重载<<
	cout << p1 << endl;
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}