#include<iostream>
#include<string>
using namespace std;

class Person {
public:
	Person(string name, int age) {
		this->m_Name = name;
		this->m_Age = age;
	}

	string m_Name;
	int m_Age;
};

// ͨ��ģ������������ݱȽ�
template<typename T>
bool myCompare(T& a, T& b) {
	cout << "����myCompare<T>()ģ�溯��" << endl;
	if (a == b) {
		return true;
	}
	return false;
}

// ���廯ģ�溯��Person�����߱�����Person��������һ�º���
// �ú���ΪmyCompare<T>()ģ�溯�����ػ����ͣ�������ΪPerson����ʱ��ֻ����һ�º�����ʵ��
template<> bool myCompare<Person>(Person& a, Person& b) {
	cout << "�����ػ���myCompare<T>()ģ�溯��" << endl;
	if (a.m_Age == b.m_Age && a.m_Name == b.m_Name) {
		return true;
	}
	return false;
}


void test01() {
	Person p1("Tom", 19);
	Person p2("Jerry", 20);
	int p3 = 1;
	int p4 = 1;
	bool ret = myCompare(p1, p2);		// ����Person����ֱ�ӶԱȣ�����1�����أ�����2�����廯
	if (ret) {
		cout << "p1��p2��ȣ�" << endl;
	}
	else {
		cout << "p1��p2����ȣ�" << endl;
	}
	bool ret2 = myCompare(p3, p4);
	if (ret2) {
		cout << "p1��p2��ȣ�" << endl;
	}
	else {
		cout << "p1��p2����ȣ�" << endl;
	}
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}