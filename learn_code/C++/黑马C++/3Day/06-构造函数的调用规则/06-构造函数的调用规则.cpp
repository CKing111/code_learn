#include<iostream>

using namespace std;

class MyClass {
public:
	// ϵͳ��Ĭ���ṩ����������Ĭ�Ϲ��캯����Ĭ�Ͽ�����������������
	//MyClass() { cout << "Ĭ�Ϲ��캯����" << endl; }
	//MyClass(const MyClass& m) { cout << "Ĭ�Ͽ������캯����" << endl; }
	//~MyClass() { cout << "Ĭ������������" << endl; }


	MyClass(int a) { cout << "�вι��캯����" << endl; }
	int m_A;
};


// ���캯�����ù���
// 1.����Ѿ��ṩ�вι��캯������ôϵͳ�Ͳ������ṩĬ���޲ι��캯����Ҫ��ʹ��Ĭ�Ϲ��캯����ֻ���Լ�д
//		����ϵͳ���ǻ��ṩĬ�ϵĿ�������
void test01() {
	//MyClass c1;		// ʧ�ܣ�classû��Ĭ�Ϲ��캯����ֻ���вι��캯��
	MyClass c2(100);	// �вι���
	c2.m_A = 100;
	MyClass c3(c2);		// Ĭ�Ͽ�������
	cout << "c3ͨ��Ĭ�Ͽ������쿽��c2��m_Aֵ��" << c3.m_A << endl;
}

// 2.�������ṩ�˿������죬ϵͳ�Ͳ����ṩ����Ĭ�Ͽ�������
class MyClass2 { 
public: 
	MyClass2(const MyClass2& m) {}// �Զ��忽������
};
void test02() {
	//MyClass2 c1;	// ʧ�ܣ���Ĭ�Ϲ���
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}