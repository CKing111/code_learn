#include<iostream>

using namespace std;

// �����أ�chong����
//	1.C++�У������������ظ�
//	2.����Ҫ��ͬһ���������£���������ͬ�Żᷢ������
//	3.����������������������ͻ���˳��ͬ�ſ���
//	4.��������ֵ��������Ϊ��������������
//	5.ע��: �������غ�Ĭ�ϲ���һ��ʹ�ã���Ҫ����ע�����������Ĳ���
//	6.���ÿ�����Ϊ�������ݣ�const����Ҳ����


// ȫ��������
void func() {
	cout << "�޲�����func" << endl;
};			
void func(int a) { 
	cout << "�в�����func(int a)" << endl;
};	
void func(double a) {
	cout << "�в�����func(double a)" << endl;
}
void func(double a, int b) {
	cout << "�в�����func(double a,int b)" << endl;
}
void func(int a, double b) {
	cout << "�в�����func(int a,double b)" << endl;
}
//�������Ͳ�ͬ��ʧ�ܣ����ֶ�����
//int func(int a, double b) {
	//cout << "�в�����func(int a,double b)" << endl;
//}
// class������
class Person {
public:
	void func() {
		cout << "Person���е��޲�����func����" << endl;
	}
};
void test01() {
	func();
	func(1);
	func(1.1);
	func(1.1, 1);
	func(1, 1.1);
	Person p1;
	p1.func();
}

// ע��: �������غ�Ĭ�ϲ���һ��ʹ�ã���Ҫ����ע�����������Ĳ���
void func2(int a) {};
void func2(int a, int b = 10) {};
// void test02() { func2(1) };		// �������أ��޷���ȷ��Դ��������

// ���õ����ذ汾
void func3(int& a) { cout << "���ð汾func(int  a)" << endl; };
void func3(const int& a) { cout << "const����func(const int & a )" << endl; };
void test03() {
	int a = 10;
	func3(a);		// ����
	func3(10);		// һ�����ò����Բ����ԣ����ÿռ䲻�Ϸ��� ��const���������
}


int main() {
	//test01();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}