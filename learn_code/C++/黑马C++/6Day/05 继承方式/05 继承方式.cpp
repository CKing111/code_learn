#include<iostream>

using namespace std;
// 继承方式：
// 公共继承：class 子类 ：public 父类{}--------不可访问父类私有，其他不变
// 保护继承：class 子类 ：protected 父类{}-----不可访问父类私有，其他变保护
// 私有继承：class 子类 ：private 父类{}-------不可访问父类私有，其他变私有

// 父类
class Base1 {
public:
	int m_A;
protected:
	int m_B;
private:
	int m_C;
};

// 公共继承
class Son1 : public Base1 {
public:
	void func() {
		m_A = 100;			// pub -> pub
		m_B = 200;			// pro -> pro
		//m_C = 300;		// 不可访问
	}
};
void test01() {
	Son1 s;
	s.m_A = 105;		// 可访问
}
// 保护继承
class Son2 : protected Base1 {
public:
	void func() {
		m_A = 100;			// public -> protected
		m_B = 200;			// pro -> pro
		//m_C = 300;		// 不可访问
	}
};
void test02() {
	Son2 s;
	//s.m_A = 105;		// 不可访问
}
// 私有继承
class Son3 : protected Base1 {
public:
	void func() {
		m_A = 100;			// public -> private
		m_B = 200;			// pro ->private
		//m_C = 300;		// 不可访问
	}
};
void test03() {
	Son2 s;
	//s.m_A = 105;		// 不可访问
	//s.m_B = 105;		// 不可访问
	//s.m_C = 105;		// 不可访问
}

int main() {
	return EXIT_SUCCESS;
}