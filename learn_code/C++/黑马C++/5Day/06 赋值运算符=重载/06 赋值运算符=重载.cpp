#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<string>
//#include<>
using namespace std;

class Person {
public:
	Person(const char * name, int age) {
		this->m_Name = new char[strlen(name) + 1];		// new���ٿռ�
		strcpy(this->m_Name, name);						// �ռ丳ֵ
		this->m_Age = age;
	}
	
	// ��ԭ�е�����=�Ż����ϸ��ģ�ʵ�ֶ�ָ�����Ĳ���
	Person& operator=(const Person& p){
		// �ж��Ƿ��ͷ�����ָ��
		if (this->m_Name != NULL) {
			delete[] this->m_Name;
			this->m_Name = NULL;
		}

		this->m_Name = new char[strlen(p.m_Name) + 1];
		strcpy(this->m_Name, p.m_Name);

		this->m_Age = p.m_Age;

		return *this;
	}

	// ��������
	Person(const Person& p) {
		this->m_Name = new char[strlen(p.m_Name) + 1];
		strcpy(this->m_Name, p.m_Name);
		this->m_Age = p.m_Age;
	}

	~Person() {
		delete[] this->m_Name;
		this->m_Name = NULL;
	}

	const char* GetName() const {
		return m_Name;
	}

	int GetAge() const {
		return m_Age;
	}

private:
	char* m_Name;
	int m_Age;
};

void test01() {
	Person p1("Tom", 18);
	Person p2("Jerry", 19);

	// ���صȺţ�
	// 1.ʵ��ָ�����ĸ�ֵ
	p1 = p2;

	cout << "p1��������" << p1.GetName() << "�� ���䣺" << p1.GetAge() << endl;
	cout << "p2��������" << p2.GetName() << "�� ���䣺" << p2.GetAge() << endl;

	// 2.ʵ�������Ⱥ�
	Person p3(" ",21);
	p3 = p1 = p2;		// �������أ���������
	cout << "p3��������" << p3.GetName() << "�� ���䣺" << p3.GetAge() << endl;

	Person p4(p3);		// ��������
	cout << "p4��������" << p4.GetName() << "�� ���䣺" << p4.GetAge() << endl;
}
int main() {
	test01();


	system("pause");
	return EXIT_SUCCESS;
}