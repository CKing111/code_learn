#include<iostream>

using namespace std;

// 存在继承关系时，切父类为类模板
// 此时，子类在创建时候，必须给定父类模板T的类型，才能分配父类的内存

template<class T>
class Base {
public:

	T m_A;
};

template<class T1, class T2>
class Son : public Base<T2> {
public:

	T1 m_B;
};


void test01(){
	Son<int, double> s;
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}