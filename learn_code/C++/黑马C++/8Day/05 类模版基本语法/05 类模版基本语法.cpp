# include<iostream>
# include<string>
using namespace std;

// 类模版
// 1. template下面紧跟着的内容是类，类中成员类型和函数为T
// 2. 类模版类型可以有默认参数


// 什么是泛型编程：类型参数化


template<class NAMETYPE, class AGETYPE = int>
class Person {
public:
	Person(NAMETYPE name, AGETYPE age) {
		this->m_Age = age;
		this->m_Name = name;
	}


	NAMETYPE m_Name;
	AGETYPE m_Age;
};


void test01() {
	//Person p1("Tom", 19);		// 失败，对于类模版不可以使用自动类型推导
	Person<string, int> p1("Tom", 19);	// 需要显示指定类型
	cout << "标准类模版，姓名：" << p1.m_Name << ", 年龄：" << p1.m_Age << endl;
	
	Person<string> p2("Jerry", 20);	// 需要显示指定类型
	cout << "默认参数类模版，姓名：" << p2.m_Name << ", 年龄：" << p2.m_Age << endl;

}


int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}