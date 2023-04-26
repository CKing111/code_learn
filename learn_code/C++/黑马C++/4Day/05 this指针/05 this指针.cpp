#include <iostream>;

using namespace std;


class Person {
public:
	Person(int age) {
		m_Age = age;		//����
		//age = age;			//����
		//this->age = age;		//������
	}
	void showAge();	// �������

	Person& addAge(Person& p);	// ������䣬���ض���

	Person addAge_num(Person& p);

	int  m_Age;
	//int age;
};

void Person::showAge() {
	cout << "���䣺" << this->m_Age << endl;
}

// ����������������뵱ǰ���ú�����������
Person& Person::addAge(Person& p) {
	this->m_Age += p.m_Age;				// thisָ�����ú�����ʵ��
	return *this;						// �����ã�����Person��
}

// ������������������ú��������������ӣ�����ֵ
Person Person::addAge_num(Person& p) {
	this->m_Age += p.m_Age;
	return *this;
}
void test01() {
	Person p1(18);

	cout << "p1�����䣺" << p1.m_Age<<endl;
	p1.showAge();		// 18

	Person p2(20);
	p2.showAge();		// 20

	cout << "p1��p2������֮�ͣ�" << p1.addAge(p2).m_Age << endl;	// 38
	// ��ʽ��̣���������Person��
	p1.m_Age = 18;
	cout << "��ʽ��̣�" << p1.addAge(p2).addAge(p2).addAge(p2).m_Age << endl; // 98
	p1.showAge();	// 98

	p1.m_Age = 18;
	p2.m_Age = 20;
	//����ֵ
	cout << "����ֵ��ӣ�" << p1.addAge_num(p2).addAge_num(p2).addAge_num(p2).m_Age << endl; // 98
	p1.showAge();	// 98
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}