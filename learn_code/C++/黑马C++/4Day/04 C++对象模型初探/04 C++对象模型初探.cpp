#include <iostream>;

using namespace std;

#pragma pack(show) // 对齐模数

// 知识点1: 类大小的计算
// 知识点2： this指针
class Person {
public:
	int m_A;		// 大小4， 成员属性，属于Person类的大小中
	double m_C;		// 大小8
	void func() {		// 默认this指针
		m_A = 100;
	}; // 成员函数不属于类的大小中，

	static int m_B; // 静态成员变量，也不属于类大小中

	static void func2() {};	// 静态成员函数也不算类大小
};

int Person::m_B = 0;


void test01() {
	cout << sizeof(Person) << endl;		// 当person为空时，占内存为1
	// 空类也是可以转化为实例的，有自己的地址
	// Person p[10] : &p[1] != &p[0]

	// this指针指向被调用的成员函数所属的对象
	Person p1;
	p1.func();	// func( this -> p1)

	Person p2;
	p2.func();	// func( this -> p2)
}
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}
