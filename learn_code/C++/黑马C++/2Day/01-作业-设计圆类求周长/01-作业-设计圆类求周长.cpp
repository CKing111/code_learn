#include<iostream>

using namespace std;

/*
	��Ŀ���ܣ�
		1.	���һ���࣬��Բ���ܳ���
	��Ŀ˼·��
		ͨ���������˼·�����һ�����ɶ���Բ�����������ܳ�
		�ܳ���ʽ�� 2 * pi * r
*/
// 1.����pi����,��const���#define
const double pi = 3.14;
// 2.���һ����Բ�ܳ�����
class Circle {
public: // ����Ȩ��
// ��������
	int m_R;	// �뾶,��Ա����

// ������Ա����
	// ���ܳ���Ա����
	double calculateZC() {
		return 2 * pi * m_R;
	}

	// ���ð뾶�ĳ�Ա����
	// ��Ա�����ǿ����޸Ĺ�����Ա����
	void setR(int r) {
		m_R = r;
	}
};

// 
void test01() {
	// ͨ���ഴ��һ��Բ
	Circle c1;  // Բ������
	// 1.ֱ�ӳ�Ա��ֵ����
	//c1.m_R = 10;
	// 2.ͨ����Ա��������Ӹ�Բ��ư뾶
	c1.setR(10);
	// ��ӡc1Բ�ܳ����
	cout << "c1���ܳ��� " << c1.calculateZC() << endl;
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}