#include <iostream>

using namespace std;

class Person {
public:
	int m_A;

};

void test01() {
	cout << sizeof(Person) << endl;		// �����СΪ1
	// ����Ҳ����ʵ����Ϊ����ӵ���Լ���һ�޶��ĵ�ַ
	// Person p[10]  : &p[0]  !=  &p[1] 
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}