#include <iostream>

using namespace std;

class Person {
	// ��Ԫ
	friend ostream& operator<<(ostream& cout, Person& p1);
public:
	Person() {};									// Ĭ�Ϲ���
	Person(int a, int b) {
		this->m_A = a;
		this->m_B = b;
	}												// �вι���

	// ��Ա����<<���㷨����
	// ���ʣ� p1.operator<< (cout) : p1<<cout
	// ������ϰ�ߣ����<<����������ó�Ա��������
	//void operator<<(ostream& cout) {
	//	
	//}
private:
	int m_A;
	int m_B;

};

// ȫ�ֺ���<<���㷨����
ostream& operator<<(ostream& cout,Person& p1) {
	cout << "m_A = " << p1.m_A << ", m_B = " << p1.m_B;
	return cout;
}

void test01() {
	Person p1(10, 10);

	//cout << " p1��m_A = " << p1.m_A << ", p1��m_B = " << p1.m_B << endl;

	// Ŀ�ģ���ͨ��cout<<ֱ�Ӵ�ӡPerson�Ĳ���
	// ����<<
	cout << p1 << endl;
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}