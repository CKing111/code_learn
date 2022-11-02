#include<iostream>

using namespace std;

class MyClass {
public:
	// 系统会默认提供三个函数：默认构造函数、默认拷贝函数、析构函数
	//MyClass() { cout << "默认构造函数！" << endl; }
	//MyClass(const MyClass& m) { cout << "默认拷贝构造函数！" << endl; }
	//~MyClass() { cout << "默认析构函数！" << endl; }


	MyClass(int a) { cout << "有参构造函数！" << endl; }
	int m_A;
};


// 构造函数调用规则
// 1.如果已经提供有参构造函数，那么系统就不会再提供默认无参构造函数，要想使用默认构造函数，只能自己写
//		但是系统还是会提供默认的拷贝构造
void test01() {
	//MyClass c1;		// 失败，class没有默认构造函数，只有有参构造函数
	MyClass c2(100);	// 有参构造
	c2.m_A = 100;
	MyClass c3(c2);		// 默认拷贝构造
	cout << "c3通过默认拷贝构造拷贝c2的m_A值：" << c3.m_A << endl;
}

// 2.当我们提供了拷贝构造，系统就不会提供其他默认拷贝构造
class MyClass2 { 
public: 
	MyClass2(const MyClass2& m) {}// 自定义拷贝构造
};
void test02() {
	//MyClass2 c1;	// 失败，无默认构造
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}