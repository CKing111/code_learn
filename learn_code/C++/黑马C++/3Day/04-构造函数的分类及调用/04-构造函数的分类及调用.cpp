#include<iostream>

using namespace std;

// 一、构造函数分类
//	1.按照参数分类
//	（1）无参构造函数：默认构造函数；
//	（2）有参构造函数；
//  2.按照类型进行分类
//	（1）普通构造函数；
//	（2）拷贝构造函数；


class Person {
public:  // 构造和析构函数必须放在public下才可以调用
	// 普通构造函数
	Person() { cout << "默认构造函数调用！" << endl; }		// 默认构造函数（无参）
	Person(int a) { 
		m_Age = a;			// 参数构造函数可传参
		cout << "有参构造函数调用！" << endl; 
	}	// 有参构造函数
	// 拷贝构造函数, 固定格式
	// 作用就是赋值类的内容
	// 拷贝函数必须加const，不允许拷贝过程修改内容
	Person(const Person& p) { 
		m_Age = p.m_Age;		// 赋值拷贝对象的公共参数
		cout << "拷贝构造函数调用！" << endl; 
	}

	~Person() { cout << "析构函数调用！" << endl; }			// 析构函数

	int m_Age;		// 公共参数
};


void test01() {
	// 1.括号法调用
	Person p1;		// 默认构造函数，不可加“（）”，加上该行表示函数声明，并不会调用构造
	Person p2(10);	// 有参构造函数，传参
	Person p3(p2);	// 拷贝构造函数

	cout << "p3的年龄：" << p3.m_Age << endl;
}

void test02() {
	// 2.显示法调用，初始化对象放在右值
	Person p1 = Person(100);	// 有参函数构造，等价于Person p1(Person(100))
	Person p2 = Person(p1);		// 拷贝函数调用，等价于Person p2(Person(p1))
	Person(100); // 表示匿名对象，编译器在执行该行代码后会马上释放这个对象

	// 不能用拷贝构造函数初始化匿名对象
	//Person(p2);		// 失败，编译器会理解为：Person p2,重复声明，可放在右值
}

void test03() {
	// 3.等号法调用
	Person p1 = 100;		// 隐式类型转换，等价于Person p1 = Person(100) 
	Person p2 = p1;			// 拷贝构造，等价于Person p2 = Person(p1)
}
int main() {
	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}