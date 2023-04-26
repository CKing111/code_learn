#include<iostream>
#include<string>
#include"myArray.hpp"

using namespace std;

// ��ӡ����
void printIntArr(MyArray<int>& myArr) {
	for (int i = 0; i < myArr.getSize(); i++) {
		cout << myArr[i] << " ";
	}
	cout << endl;
}

// ����int��������
void test01() {
	MyArray<int> myIntArr(100);
	for (int i = 0; i < 10; i++) {
		myIntArr.pushBack(i + 100);
	}
	// ��ӡ����
	printIntArr(myIntArr);
}

// �����Զ�����������
class Person {
public:
	Person() {};
	Person(string name, int age) :m_Name(name), m_Age(age) {}		// �б��ʼ��
	string m_Name;
	int m_Age;
};
void printPerson(MyArray<Person>& myArr) {
	for (int i = 0; i < myArr.getSize(); i++) {
		cout << "������" << myArr[i].m_Name << ", ���䣺" << myArr[i].m_Age << endl;
	}
}
void test02() {
	// �����Զ�����������Person
	MyArray <Person> personArr(10);
	Person p1("�����1",599);
	Person p2("�����2",19);
	Person p3("�����3",29);
	Person p4("�����4",39);
	Person p5("�����5",49);

	personArr.pushBack(p1);
	personArr.pushBack(p2);
	personArr.pushBack(p3);
	personArr.pushBack(p4);
	personArr.pushBack(p5);
	
	// ��ӡ
	printPerson(personArr);
}

int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}