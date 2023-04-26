#include<iostream>

using namespace std;

// �̳з�ʽ��
// �����̳У�class ���� ��public ����{}--------���ɷ��ʸ���˽�У���������
// �����̳У�class ���� ��protected ����{}-----���ɷ��ʸ���˽�У������䱣��
// ˽�м̳У�class ���� ��private ����{}-------���ɷ��ʸ���˽�У�������˽��

// �����е�˽��Ҳ���̳У�ֻ�Ǳ����أ�����ͨ�����������鿴��cl /d1 reportSingleClassLayout+���� �ļ�����

// �̳��У��ȵ��ø��๹�죬��ʹ�����๹�죬������˳���෴
// ���಻��̳� �����еĹ������������

// ����
class Base {
public:
	Base() {
		cout << "����Base�е�Ĭ�Ϲ��캯��" << endl;
	}
	~Base() {
		cout << "����Base�е���������" << endl;
	}

	int m_A;
protected:
	int m_B;
private:
	int m_C;
};

class Son : public Base {
public:	
	Son() {
		cout << "����Son�е�Ĭ�Ϲ��캯��" << endl;
	}
	~Son() {
		cout << "����Son�е���������" << endl;
	}
};

void test01() {
	//Base b;

	Son s;
	/*����Base�е�Ĭ�Ϲ��캯��
	����Son�е�Ĭ�Ϲ��캯��
	����Son�е���������
	����Base�е���������*/
}

class Base2 {
public:
	Base2(int a) {					// ֻ���вι��캯��
		this->m_A = 1;
	}
	int m_A;
};
// �̳��вι��캯��
class Son2 : public Base2 {				// �� ����̳�
public:
	// ��ʼ���б��﷨
	// �������ó�ʼ���б�ķ�ʽ��ʾ�������Ǹ�������Ǹ����캯��
	Son2(int a) : Base2(1) {					// : �����ʼ���б��������ø�����вι���
		this->m_B = m_A;
	}
	int m_B;
};
void test02() {
	Son2 s(10);
	cout << s.m_A << s.m_B << endl;
}
int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}