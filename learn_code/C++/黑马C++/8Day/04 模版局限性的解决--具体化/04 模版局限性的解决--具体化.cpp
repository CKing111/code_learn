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

// 通过模版进行两个数据比较
template<typename T>
bool myCompare(T& a, T& b) {
	cout << "运行myCompare<T>()模版函数" << endl;
	if (a == b) {
		return true;
	}
	return false;
}

// 具体化模版函数Person，告诉编译器Person类输入走一下函数
// 该函数为myCompare<T>()模版函数的特化类型，当输入为Person类型时，只能走一下函数的实现
template<> bool myCompare<Person>(Person& a, Person& b) {
	cout << "运行特化版myCompare<T>()模版函数" << endl;
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
	bool ret = myCompare(p1, p2);		// 两个Person不能直接对比，方法1：重载，方法2：具体化
	if (ret) {
		cout << "p1与p2相等！" << endl;
	}
	else {
		cout << "p1和p2不相等！" << endl;
	}
	bool ret2 = myCompare(p3, p4);
	if (ret2) {
		cout << "p1与p2相等！" << endl;
	}
	else {
		cout << "p1和p2不相等！" << endl;
	}
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}