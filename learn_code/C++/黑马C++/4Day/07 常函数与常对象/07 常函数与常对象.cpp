#include <iostream>

using namespace std;

class Person {
public:
	void showPerson() {
		cout << this->m_A << endl;
		this->m_A = 100;	// �ɹ�
		//this = NULL  // ����
		// Person * const this
		// thisָ��ı��ʾ���һ��ָ�볣����ָ���ָ���ǲ����Ըı�ģ�ָ���ָ��ֵ���Ը�
	}

	//  ������
	// ��Ա�����������const���������������޸ĳ�Ա����
	// ��������mutable�����ĳ�Ա����
	void showPerson_const() const {		// ������
		cout << this->m_A << endl;
		//this->m_A = 100;	// ʧ�ܣ�const Person * const this
		this->m_B = 100;	// �ɹ�
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

// ������
// ��������Բ���mutable��Ա�������������޸ķ�mutable��Ա������
void test02(){
	const Person p2;	// ������
	//p2.m_A = 100;		// ʧ��
	p2.m_B = 100;		// �ɹ�

	p2.showPerson_const();	// �ɹ���������ֻ�ܵ��ó�����
	//p2.showPerson2();		// ʧ�ܣ��������ǲ����Ե���������Ա������

}
int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}