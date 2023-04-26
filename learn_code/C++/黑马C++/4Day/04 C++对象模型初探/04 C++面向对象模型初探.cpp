#include <iostream>

using namespace std;

class Person {
public:
	int m_A;

};

void test01() {
	cout << sizeof(Person) << endl;		// 空类大小为1
	// 空类也可以实例化为对象，拥有自己独一无二的地址
	// Person p[10]  : &p[0]  !=  &p[1] 
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}