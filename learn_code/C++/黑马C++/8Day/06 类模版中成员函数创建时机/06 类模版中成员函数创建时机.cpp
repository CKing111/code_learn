#include<iostream>

using namespace std;

class Person1 {
public:
	void showPerson1() {
		cout << "Person1 show" << endl;
	}
};

class Person2 {
public:
	void showPerson2() {
		cout << "Person2 show" << endl;
	}
};


// 类模版中的成员函数并不是一开始就创建出来的，
// 而是在运行阶段才创建出来的

template<class T>
class myClass {
public:
	// 只有运行后才会只到对象所包含的成员及成员函数

	void func1() {
		obj.showPerson1();
	}

	void func2() {
		obj.showPerson2();
	}
	T obj;		
};

void test01() {
	myClass<Person1> p1;
	p1.func1();
	//p1.func2();		// 失败，推导的T类型不支持person2的类型
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}