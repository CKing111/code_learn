#include<iostream>

using namespace std;


class Base {
public:
	Base() {
		this->m_A = 100;
	}
	void func() {
		cout << "Base�е�func��" << endl;
	}

	void func(int a) {
		cout << "Base�е�func��int a��:" << a << endl;
	};
	int m_A;
};

// �����д���ͬ����Ա���������þͽ�ԭ�����Ӻ�
class Son : public Base {
public:
	Son() {
		this->m_A = 200;
	}
	void func() {
		cout << "Son�е�func��" << endl;
	}
	int m_A;
};


void test01() {
	Son m;
	cout << m.m_A << endl;				// �ͽ�ԭ����������ֵ
	cout << "Base�е�m_A��(m.Base::m_A )" << m.Base::m_A << endl;		// �������ʸ��࣬����������

	m.func();		// �ͽ�
	m.Base::func();		// ����

	//m.func(10);			// ����ͬ���ĳ�Ա�������͸��ݾͽ�ԭ����������ε������е���������
	m.Base::func(10);		// ��������Ի�ȡ�������ذ汾
}

/*
class Son       size(8):
		+---
 0      | +--- (base class Base)
 0      | | m_A
		| +---
 4      | m_A
		+---N
*/
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;


}