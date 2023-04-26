# include<iostream>
# include<string>
using namespace std;

template<class NAMETYPE, class AGETYPE >
class Person {
public:
	Person(NAMETYPE name, AGETYPE age) {
		this->m_Age = age;
		this->m_Name = name;
	}


	NAMETYPE m_Name;
	AGETYPE m_Age;
};

// 类模版生成的对象作为函数参数有以下几种方式：
// 1. 制定传入类型
void doWork1(Person<string, int>& p) {
	cout << "方法1，制定传入类型，姓名：" << p.m_Name << ", 年龄：" << p.m_Age << endl;
}

// 2. 参数模版化
template<class T1, class T2>		// 函数模版
void doWork2(Person<T1, T2>& p) {
	cout << "方法2，参数模版化，姓名：" << p.m_Name << ", 年龄：" << p.m_Age << endl;
	cout << "T1 type: " << typeid(T1).name() << ", T2 type: " << typeid(T2).name() << endl;
}

// 3. 整个类模版化
template<class T>
void doWork3(T& p) {
	cout << "方法3，整个类模版化，姓名：" << p.m_Name << ", 年龄：" << p.m_Age << endl;
	cout << "T type: " << typeid(T).name() << endl;
}

void test01() {
	Person<string, int> p1("Tom", 29);

	// 1.
	doWork1(p1);
	doWork2(p1);
	doWork3(p1);
/*
方法1，制定传入类型，姓名：Tom, 年龄：29
方法2，参数模版化，姓名：Tom, 年龄：29
T1 type: class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> >, T2 type: int
方法3，整个类模版化，姓名：Tom, 年龄：29
T type: class Person<class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> >,int>
请按任意键继续. . .
*/
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}