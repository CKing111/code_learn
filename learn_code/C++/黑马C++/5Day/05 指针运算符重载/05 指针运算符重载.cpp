#include<iostream>

using namespace std;

class Person {
public:
	Person(int age) {
		this->m_Age = age;
		cout << "Person有参构造函数" << endl;
	}

	void showAge() {
		cout << "年龄为：" << this->m_Age << endl;
	}

	~Person() {
		cout << "Person的析构函数" << endl;
	}

private:
	int m_Age;
};

// 智能指针 （类）
// 用来托管new出来的指针的释放
class SmartPointer {
public:
	SmartPointer(Person* person) {			// 有参构造
		this->person = person;
		cout << "SmartPointer的有参构造" << endl;
	}

	// 重载指针运算符，使构造的类像指针一样操作
	// 1.重载->
	Person* operator->() {			// this = Person* person
		cout << "->重载" << endl;
		return this->person;
	}
	// 重载（*）
	Person & operator*() {		// 返回引用，代表返回本体，不要再copy副本
		cout << "(*)重载" << endl;
		return *this->person;	// 
	}

	~SmartPointer() {						// 析构函数，自动释放
		cout << "SmartPointer析构函数，释放指针" << endl;
		if (this->person != NULL) {
			delete this->person;
			this->person = NULL;
		}
	}
private:
	Person* person;
};
void test01() {

	// 1.指针构造
	//Person* p = new Person(18);			// 开辟指针
	//delete p;							// 释放指针
	//p->showAge();		// 指针操作
	//(*p).showAge();	// 指针解引用
	//  2.智能指针构造类
	SmartPointer sp = SmartPointer(new Person(18));		// 有参构造
	// 设置指针操作
	// 重载->指针符号
	sp->showAge();		// 返回Person*，理论上需要sp->->showAge()，编译器辅助省略了
	// 重载解引用（*）
	(*sp).showAge();
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}