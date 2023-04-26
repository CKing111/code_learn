#include<iostream>

using namespace std;

class MyInter {
	friend ostream& operator<<(ostream& cout, MyInter& myint);
public:
	MyInter() {
		this->m_Num = 0;
	}
	// 成员函数重载前置++运算符
	// 返回类对象引用，方便进行递增运算，目的是一直对本体进行计算
	MyInter& operator++() {
		// 先++
		m_Num++;
		// 后返回
		return *this;
	}
	// 成员函数重载后置++运算符
	MyInter operator++(int) {
		// 先返回
		MyInter temp = *this;	// 保存旧址
		// 值后++
		++this->m_Num;
		// 返回旧址
		return temp;
	}
private:
	int m_Num;
};

ostream& operator<<(ostream& cout, MyInter& myint) {
	cout << myint.m_Num;
	return cout;
}


void test01() {
	MyInter myint;
	//cout << myint << endl;
	cout << "重载前置++：" << ++myint << endl;
	cout << myint << endl;
}

void test02() {
	MyInter myint2;
	//cout << "重载前置++：" << ++myint << endl;
	MyInter m2 = myint2++;
	cout << m2  << endl;
	m2++;
	cout << m2 << endl;
}
int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}