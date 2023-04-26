#include<iostream>

using namespace std;

// 继承方式：
// 公共继承：class 子类 ：public 父类{}--------不可访问父类私有，其他不变
// 保护继承：class 子类 ：protected 父类{}-----不可访问父类私有，其他变保护
// 私有继承：class 子类 ：private 父类{}-------不可访问父类私有，其他变私有

// 父类中的私有也被继承，只是被隐藏，可以通过其他方法查看（cl /d1 reportSingleClassLayout+类名 文件名）

// 继承中：先调用父类构造，再使用子类构造，析构的顺序相反
// 子类不会继承 父类中的构造和析构函数

// 父类
class Base {
public:
	Base() {
		cout << "父类Base中的默认构造函数" << endl;
	}
	~Base() {
		cout << "父类Base中的析构函数" << endl;
	}

	int m_A;
protected:
	int m_B;
private:
	int m_C;
};

class Son : public Base {
public:	
	Son() {
		cout << "父类Son中的默认构造函数" << endl;
	}
	~Son() {
		cout << "父类Son中的析构函数" << endl;
	}
};

void test01() {
	//Base b;

	Son s;
	/*父类Base中的默认构造函数
	父类Son中的默认构造函数
	父类Son中的析构函数
	父类Base中的析构函数*/
}

class Base2 {
public:
	Base2(int a) {					// 只有有参构造函数
		this->m_A = 1;
	}
	int m_A;
};
// 继承有参构造函数
class Son2 : public Base2 {				// ： 代表继承
public:
	// 初始化列表语法
	// 可以利用初始化列表的方式显示出调用那个父类的那个构造函数
	Son2(int a) : Base2(1) {					// : 代表初始化列表，引导调用父类的有参构造
		this->m_B = m_A;
	}
	int m_B;
};
void test02() {
	Son2 s(10);
	cout << s.m_A << s.m_B << endl;
}
int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}