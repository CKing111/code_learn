#include<iostream>

using namespace std;

/*
	1.1 ��̬��Ա����
		1.1.1 ����׶η����ڴ�
		1.1.2 ���ж���������
		1.1.3 ͨ��������������
		1.1.4 ��Ȩ�޿���
		1.1.5 ���������������ʼ��
	1.2 ��̬��Ա����
		1.2.1 ���Է��ʾ�̬��Ա�����������Է�����ͨ��Ա����
		1.2.2 ��ͨ��Ա���������Է���
		1.2.3 ��̬��Ա����Ҳ��Ȩ��
		1.2.4 ����ͨ�������������� 
*/



class Person {
public:
	Person() {}
	int m_A;
	static int m_Age;		// static���Ǿ�̬��Ա�������Ṳ������
	// ��̬��Ա��������������������������г�ʼ��

	// ��̬��Ա����
	// �����Է�����ͨ��Ա�������޷���ȷ����ֵ��������һ��������
	// �������ʹ���ľ�̬��Ա����������Ҫ����������Դ�Ǹ�������
	static void func() {
		//m_A = 10;		// ʧ�ܣ���ͨ��Ա����
		m_Age = 10;		// �ɹ��������Ա����
		cout << "��̬��Ա�������ã�" << endl;
	}

	// ��̬��Ա����Ҳ����Ȩ�޵�
private:
	static int m_Other;
	static void func2() {
		cout << "˽�пռ䡢��̬��Ա����func2���ã�" << endl;
	}
};

int Person::m_Age = 0;		// �����ʼ��ʵ��
int Person::m_Other = 10;	// ˽��Ȩ����������Գ�ʼ��

void test01() {
	Person p1;
	p1.m_Age = 10;
	cout << "p1��Age��" << p1.m_Age << endl;

	Person p2;
	p2.m_Age = 20;
	// 1.ͨ�������������
	cout << "p1��Age��" << p1.m_Age << endl;
	cout << "p2��Age��" << p2.m_Age << endl;

	// 2.ͨ��������������
	cout << "ͨ���������ʾ�̬����m_Age��" << Person::m_Age << endl;
	//cout << "m_Other��" << Person::m_Other << endl;  // ʧ�ܣ�˽��Ȩ���������޷�����

	// ��̬��Ա��������
	p1.func();
	p2.func();
	Person::func();

	//Person::func2();		// ʧ�ܣ�˽�пռ�
}


int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}