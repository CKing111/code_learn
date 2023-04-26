#include<iostream>

using namespace std;
// �̳з�ʽ��
// �����̳У�class ���� ��public ����{}--------���ɷ��ʸ���˽�У���������
// �����̳У�class ���� ��protected ����{}-----���ɷ��ʸ���˽�У������䱣��
// ˽�м̳У�class ���� ��private ����{}-------���ɷ��ʸ���˽�У�������˽��

// ����
class Base1 {
public:
	int m_A;
protected:
	int m_B;
private:
	int m_C;
};

// �����̳�
class Son1 : public Base1 {
public:
	void func() {
		m_A = 100;			// pub -> pub
		m_B = 200;			// pro -> pro
		//m_C = 300;		// ���ɷ���
	}
};
void test01() {
	Son1 s;
	s.m_A = 105;		// �ɷ���
}
// �����̳�
class Son2 : protected Base1 {
public:
	void func() {
		m_A = 100;			// public -> protected
		m_B = 200;			// pro -> pro
		//m_C = 300;		// ���ɷ���
	}
};
void test02() {
	Son2 s;
	//s.m_A = 105;		// ���ɷ���
}
// ˽�м̳�
class Son3 : protected Base1 {
public:
	void func() {
		m_A = 100;			// public -> private
		m_B = 200;			// pro ->private
		//m_C = 300;		// ���ɷ���
	}
};
void test03() {
	Son2 s;
	//s.m_A = 105;		// ���ɷ���
	//s.m_B = 105;		// ���ɷ���
	//s.m_C = 105;		// ���ɷ���
}

int main() {
	return EXIT_SUCCESS;
}