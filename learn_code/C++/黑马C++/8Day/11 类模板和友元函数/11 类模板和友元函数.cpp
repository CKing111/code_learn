#include<iostream>
#include<string>
using namespace std;

// 要实现友元函数作为模版并且在类外实现（方法2），需要四步：
// 第一步：告诉编译器Person类，先不报错
template<class T1, class T2> class Person;
// 第二步：告诉编译器，有一个函数模版声明
template<class T1, class T2> void printPerson2(Person<T1, T2>& p);

// 将模版的声明和实现放在一起
template<class T1, class T2> void printPerson3(Person<T1, T2>& p) {
	cout << "类外实现2，姓名： " << p.m_Name << ", 年龄： " << p.m_Age << endl;
}

template<class T1, class T2>
class Person {
	// 1.全局函数配合友元，做类内实现
	friend void printPerson(Person<T1, T2>& p) {
		cout << "类内实现，姓名： " << p.m_Name << ", 年龄： " << p.m_Age << endl;
	}
	// 2.全局函数配合友元做类外实现
// 第三步: 声明模版函数为友元,且需要空集<>声明是模版
	friend void printPerson2<>(Person<T1, T2>& p);

	// 3. 全局函数配合友元做类外实现,将模版实现和声明放在一起
	friend void printPerson3<>(Person<T1, T2>& p);
public:
	Person(T1 name, T2 age) {
		this->m_Age = age;
		this->m_Name = age;
	}
private:
	T1 m_Name;
	T2 m_Age;
};

// 第四步，类外实现
template<class T1, class T2>
void printPerson2(Person<T1, T2>& p) {
	cout << "类外实现1，姓名： " << p.m_Name << ", 年龄： " << p.m_Age << endl;
}

void test01() {
	Person<string, int> p("Tom", 29);

	printPerson(p);
	printPerson2(p);
	printPerson3(p);

}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}