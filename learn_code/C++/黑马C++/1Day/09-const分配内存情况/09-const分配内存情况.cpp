#include<iostream>
#include<string>
using namespace std;

// ֻҪ�з����ڴ棬�Ϳ�����ָ���������
// 1.const�����ڴ棬ȡ��ַʱ�������ʱ�ڴ�
// 2.extern ���������const���������ڴ�
void test01() {
	const int m_A = 10;
	int* p = (int*)m_A;	// �������ʱ�ڴ棬��ԭʼ���ݲ���ı�
	//*p = 1000;
	//cout << "m_A = " << m_A << ", *p = " << *p << endl;
}
// 3.��ͨ�������Գ�ʼ��const����
void test02() {
	int a = 10;
	const int b = a; // ������ʼ��
	// ������ڴ棬��ʹ��ָ������޸�
	int* p = (int*)&b; //��ȡָ��
	*p = 1000;

	cout << "b = " << b << endl;
}
// 4.�Զ�����������,��constҲ������ڴ�
struct Person {
	string m_Name;  //C�����ַ����������ͣ��ǵ�����
	int m_Age;		 
};
void test03(){
	const Person p1 = {};		//const��������ʼ��.
	// ������ֱ�ӸĶ�����Ҫ��ָ��
	// ָ����Ը�˵���з����ַ
	Person* p = (Person*)&p1;  //��ȡָ��
	p->m_Name = "XXXX";
	(*p).m_Age = 18;

	cout << "������" << p1.m_Name << ", ���䣺" << p1.m_Age << endl;
}
int main() {
	test01();
	//test03();
	system("pause");
	return EXIT_SUCCESS;
}