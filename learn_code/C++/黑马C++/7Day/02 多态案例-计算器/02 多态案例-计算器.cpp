#include<iostream>
#include<string>

using namespace std;

// 开发原则：开闭原则--对扩展进行开放，对修改进行关闭
//				既：我只需要提供几个base类，功能基于此进行扩展，不修改base
// 多态的好处：提高扩展性，增强组织性，可读性强
//				如果父类中有了虚函数，子类并没有重写父类的虚函数，那将毫无意义
//				如果子类不重写父类的虚函数，那多态并无作用，且会增加代码内部复杂程度
// 
// 利用多态实现计算器
// 抽象基类计算器
class AbstractCalculator {
public:
	// 设置虚函数，方便子类扩展
	//virtual int getResult() {
	//	return 0;
	//}

	// 也可以设置纯虚函数，但是不可以实例化对象
	// 有纯虚函数的类成为抽象类，是无法实例化对象的，
	// 且子类必须重写实现父类的纯虚函数，否则子类也是抽象类 
	// 只有虚函数不是抽象类
	virtual int getResult() = 0;

	int m_A;
	int m_B;
};

// 加法计算器
/*
class AddCalculator     size(16):
		+---
 0      | +--- (base class AbstractCalculator)
 0      | | {vfptr}
 8      | | m_A
12      | | m_B
		| +---
		+---

AddCalculator::$vftable@:
		| &AddCalculator_meta
		|  0
 0      | &AddCalculator::getResult
*/
class AddCalculator :public AbstractCalculator {
public:
	virtual int getResult() {
		return m_A + m_B;
	}
};

// 减法计算器
class SubCalculator :public AbstractCalculator {
public:
	virtual int getResult() {
		return m_A - m_B;
	}
};

// 乘法计算器
class MultiCalculator :public AbstractCalculator {
public:
	virtual int getResult() {
		return m_A * m_B;
	}
};

// 除法计算器
class DivisionCalculator :public AbstractCalculator {
public:
	virtual int getResult() {
		return m_A / m_B;
	}
};


void test01() {
	// 声明并使用加法计算器
	AbstractCalculator* calculator = new AddCalculator;
	calculator->m_A = 20;
	calculator->m_B = 10;
	cout <<"20+10 = " << calculator->getResult() << endl;

	// 释放更换减法计算器
	delete calculator;
	calculator = new SubCalculator;
	calculator->m_A = 20;
	calculator->m_B = 10;
	cout << "20-10 = " << calculator->getResult() << endl;
}



int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;

}