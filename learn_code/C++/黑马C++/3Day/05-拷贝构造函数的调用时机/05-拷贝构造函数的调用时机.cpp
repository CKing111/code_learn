#include<iostream>
using namespace std;


class Person {
public:  // ��������������������public�²ſ��Ե���
	// ��ͨ���캯��
	Person() { cout << "Ĭ�Ϲ��캯�����ã�" << endl; }		// Ĭ�Ϲ��캯�����޲Σ�
	Person(int a) {
		m_Age = a;			// �������캯���ɴ���
		cout << "�вι��캯�����ã�" << endl;
	}	// �вι��캯��
	// �������캯��, �̶���ʽ
	// ���þ��Ǹ�ֵ�������
	// �������������const���������������޸�����
	// �����&����Ϊ����Ϊֵ���ݣ������ɸ��ݿ�����Person���͵� p ����������ʱ������ʱ�����ǻ���ÿ������죬Ȼ�������ѭ��
	Person(const Person& p) {
		m_Age = p.m_Age;		// ��ֵ��������Ĺ�������
		cout << "�������캯�����ã�" << endl;
	}

	~Person() { cout << "�����������ã�" << endl; }			// ��������

	int m_Age;		// ��������
};

// ��������ʹ�ó�����ʹ��ʱ��
// 1.���Ѿ������õĶ�������ʼ���µĶ���
void test01() {
	Person p1;
	p1.m_Age = 100;

	Person p2(p1);
	cout << "p2�����䣺" << p2.m_Age << endl;
}

// 2. ��ֵ���ݵķ�ʽ������������ֵ��ֵ���ݲ����ԭʼ���ݽ����޸�
//  ����ֵ����ʱ����ͨ���������캯�����ݽ�ȥֵ�����ô��ݲ�����ÿ������캯��
//  ֵ���ݻ������м����ʱֵ�����ֵʹ�ÿ�����������
void doWork(Person p) {		// �������α�ʾΪ����������Person p = Person(p1)�������˿�������
	cout << "����ֵ����Person p �����䣺" << p.m_Age << endl;
}
void test02() {
	Person p1;
	p1.m_Age = 100;

	doWork(p1);
}

// 3.��ֵ�ķ�ʽ���ؾֲ�����
//  ���������÷��ؾֲ����󣬻�ı�ֵ
Person doWork2() {		// ����ֵ
	Person p1;	// Ĭ�Ϲ��캯��
	p1.m_Age = 100;
	return p1;	// �������캯��
}
void test03() {
	Person p = doWork2();	// ʹ�ú�������ֵ����Person p����
	cout << "�����������ɺ�������ֵp�����䣺" << p.m_Age << endl;
}

// vs��releaseģʽ���Զ��ֻ����룬�ܵ������þͲ��ÿ���
// test03�Ż�Ϊ��
/*
	Person p;	// ������Ĭ�Ϲ��캯��
	doWork2(p);
	void doWork2(Person &p){
		Person p1;		// ����Ĭ�Ϲ��캯��
	}
*/
int main() {
	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}