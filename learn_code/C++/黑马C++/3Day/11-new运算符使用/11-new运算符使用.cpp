#include<iostream>
#include<string>
using namespace std;


class Person {
public:
	Person() { cout << "Ĭ�Ϲ��캯�����ã�" << endl; }
	Person(int a) { cout << "�вι��캯�����ã�" << endl; }
	~Person() { cout << "�����������ã�" << endl; }

};

void test01() {
	//Person p1;				// ջ�����٣��Զ��ͷſռ�
	Person* p1 = new Person;	// �������ٿռ䣬�����Զ��ͷſռ�

	/*
		����new�����Ķ��󣬶��᷵����ͬ���͵�ָ��
		malloc ���ص���void*ָ�룬�һ���Ҫǿ��ת�����ͣ������ռ��С
		malloc�����Զ����ù��캯����new���Զ����ù��캯��
		new���������malloc�Ǻ���
		ʹ��delete������ͷŶ����ռ�
		malloc-free��new-delete
	*/
	delete p1;
}

void test02() {
	void * p = new Person;		// ��void* ָ��ȥ����new���ɵ�ָ�룬�����Զ��ͷ�
	delete p;
}

void test03() {
	// ͨ��new��������
	Person* pArray = new Person[10];	// �������٣����Զ�����10��Ĭ�Ϲ��캯��
	//delete pArray;
	// ע�⣺new��������һ�������Ĭ�Ϲ��죬����һ��Ҫ�ṩĬ�Ϲ���
	// ջ�����ٿռ䣬����ָ���вι��죬����������
	Person pArray2[2] = { Person(1),Person(2) };		// ��ʽ����

	// �ͷŶ�������
	delete[] pArray;		// ��[]����ʾϵͳ����һ����Ŀ��������飬ϵͳ���Զ�Ѱ�ҵ��ü������������ͷſռ�
}
int main() {
	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}