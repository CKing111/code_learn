#include <iostream>

using namespace std;

class Person {
public:
	Person() {};									// Ĭ�Ϲ���
	Person(int a, int b) : m_A(a), m_B(b) {}		// �вι���

	// 1.��Ա��������
	// + �����������
	//Person operator+(Person& p) {
	//	Person temp;
	//	temp.m_A = this->m_A + p.m_A;
	//	temp.m_B = this->m_B + p.m_B;
	//	return temp;
	//}
	int m_A;
	int m_B;

};

//2.ȫ�ֺ�������(person + person)
Person operator+(Person& p1, Person& p2) {
	Person temp;
	temp.m_A = p1.m_A + p2.m_A;
	temp.m_B = p1.m_B + p2.m_B;
	return temp;
}
// (person + int)
Person operator+(Person& p1, int p) {
	Person temp;
	temp.m_A = p1.m_A + p;
	temp.m_B = p1.m_B + p;
	return temp;
}


void test01(){
	Person p1(10, 10);
	Person p2(20, 20);

	//��person + person������
	Person p3 = p1 + p2;		
// 1.��Ա��������
// ���ʣ� Person p3 = p1.operator+(p2);
// 2.ȫ�ֺ�������
// ���ʣ� Person p3 = operator+(p1, p2)
	cout << "(Person + Person)����" << endl;
	cout << "p3��m_A = " << p3.m_A << endl;
	cout << "p3��m_B = " << p3.m_B << endl; 


	// (Person + int)����
	Person p4 = p1 + 10;
	cout << "(Person + int)����" << endl;
	cout << "p4��m_A = " << p4.m_A << endl;
	cout << "p4��m_B = " << p4.m_B << endl;
}

int main() {
	test01();
	
	system("pause");
	return EXIT_SUCCESS;
}