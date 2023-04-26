#include <iostream>;

using namespace std;


class Person {
public:
	Person(int age) {
		m_Age = age;		//正常
		//age = age;			//错误
		//this->age = age;		//正常，
	}
	void showAge();	// 输出年龄

	Person& addAge(Person& p);	// 添加年龄，返回对象

	Person addAge_num(Person& p);

	int  m_Age;
	//int age;
};

void Person::showAge() {
	cout << "年龄：" << this->m_Age << endl;
}

// 将输入对象的年龄加入当前调用函数对象身上
Person& Person::addAge(Person& p) {
	this->m_Age += p.m_Age;				// this指向引用函数的实例
	return *this;						// 解引用，返回Person类
}

// 将输入对象的年龄与调用函数对象的年龄相加，返回值
Person Person::addAge_num(Person& p) {
	this->m_Age += p.m_Age;
	return *this;
}
void test01() {
	Person p1(18);

	cout << "p1的年龄：" << p1.m_Age<<endl;
	p1.showAge();		// 18

	Person p2(20);
	p2.showAge();		// 20

	cout << "p1和p2的年龄之和：" << p1.addAge(p2).m_Age << endl;	// 38
	// 链式编程，返回自身Person类
	p1.m_Age = 18;
	cout << "链式编程：" << p1.addAge(p2).addAge(p2).addAge(p2).m_Age << endl; // 98
	p1.showAge();	// 98

	p1.m_Age = 18;
	p2.m_Age = 20;
	//返回值
	cout << "返回值相加：" << p1.addAge_num(p2).addAge_num(p2).addAge_num(p2).m_Age << endl; // 98
	p1.showAge();	// 98
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}