#include<iostream>

using namespace std;

class Person {
public:
	//Person(){}	// Ĭ�Ϲ��캯��
	// �вι��캯��
	// ������ʼ������
	// ����1��
	//Person(int a, int b, int c){
	//	m_A = a;
	//	m_B = b;
	//	m_C = c;
	//}
	// ����2�����ó�ʼ���б��ʼ�����ݣ����Դ���
	// ��ʽ�� ���캯��֮�� + ����1������ֵ��������2������ֵ��������������
	Person(int a, int b, int c) :m_A(a), m_B(b), m_C(c) {}
	
	// ��ʼ���б������캯����������Ĭ�ϲ���
	// �̶����������ܴ���
	Person() :m_A(10), m_B(20), m_C(30) {}


	int m_A;
	int m_B;
	int m_C;
};
void test01() {
	Person p1(10, 20, 30);

	cout << "p1��m_A: " << p1.m_A << ", m_B: " << p1.m_B << ", m_C: " << p1.m_C << endl;

	Person p2;
	cout <<"��ʼ���б�Ĭ�Ϲ��캯������ֵ��\n" << "p2��m_A: " << p2.m_A << ", m_B: " << p2.m_B << ", m_C: " << p2.m_C << endl;

}


int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;

}