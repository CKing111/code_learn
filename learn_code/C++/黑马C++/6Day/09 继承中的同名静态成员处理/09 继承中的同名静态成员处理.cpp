#include<iostream>

using namespace std;


class Base {
public:
	static int m_A;		// ��̬��Ա���������������������ʼ��������׶η����ڴ棬��������

	static void func(){
		cout << "Base�еľ�̬��Ա����func" << endl;
	}
	static void func(int a) {
		cout << "Base�еľ�̬��Ա����func" <<a<< endl;
	}
};

int Base::m_A = 10;

class Son :public Base {
public:
	static int m_A;
	static void func() {
		cout << "Son�еľ�̬��Ա����func" << endl;
	}
};

int Son::m_A = 20;


void test01() {
	Son s;
	// ͨ��class������ʾ�̬��Ա
	cout << s.m_A << endl;
	cout << "Base�е�m_A��" << s.Base::m_A << endl;
	// ͨ���������ʣ���̬��Ա����������
	cout << "ͨ����������Base�е�m_A��" << Son::m_A << endl;
	cout << "ͨ����������Base�е�m_A��" << Son::Base::m_A << endl;	// 

	s.func();
	Son::func();
	s.Base::func();
	s.Base::func(10);

	Son::Base::func();
	Son::Base::func(10);
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}
