#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<string>
//#include<>
using namespace std;

class Person {
public:
	Person(const char * name, int age) {
		this->m_Name = new char[strlen(name) + 1];		// new开辟空间
		strcpy(this->m_Name, name);						// 空间赋值
		this->m_Age = age;
	}
	
	// 在原有的重载=号基础上更改，实现对指针量的操作
	Person& operator=(const Person& p){
		// 判断是否释放现有指针
		if (this->m_Name != NULL) {
			delete[] this->m_Name;
			this->m_Name = NULL;
		}

		this->m_Name = new char[strlen(p.m_Name) + 1];
		strcpy(this->m_Name, p.m_Name);

		this->m_Age = p.m_Age;

		return *this;
	}

	// 拷贝构造
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

	// 重载等号，
	// 1.实现指针量的赋值
	p1 = p2;

	cout << "p1的姓名：" << p1.GetName() << "， 年龄：" << p1.GetAge() << endl;
	cout << "p2的姓名：" << p2.GetName() << "， 年龄：" << p2.GetAge() << endl;

	// 2.实现三连等号
	Person p3(" ",21);
	p3 = p1 = p2;		// 更改重载，返回自身
	cout << "p3的姓名：" << p3.GetName() << "， 年龄：" << p3.GetAge() << endl;

	Person p4(p3);		// 拷贝构造
	cout << "p4的姓名：" << p4.GetName() << "， 年龄：" << p4.GetAge() << endl;
}
int main() {
	test01();


	system("pause");
	return EXIT_SUCCESS;
}