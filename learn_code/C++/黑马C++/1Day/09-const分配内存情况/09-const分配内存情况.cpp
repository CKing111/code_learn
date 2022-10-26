#include<iostream>
#include<string>
using namespace std;

// 只要有分配内存，就可以用指针操作变量
// 1.const分配内存，取地址时会分配临时内存
// 2.extern 编译器会给const变量分配内存
void test01() {
	const int m_A = 10;
	int* p = (int*)m_A;	// 会分配临时内存，但原始数据不会改变
	//*p = 1000;
	//cout << "m_A = " << m_A << ", *p = " << *p << endl;
}
// 3.普通变量可以初始化const变量
void test02() {
	int a = 10;
	const int b = a; // 变量初始化
	// 会分配内存，可使用指针进行修改
	int* p = (int*)&b; //读取指针
	*p = 1000;

	cout << "b = " << b << endl;
}
// 4.自定义数据类型,加const也会分配内存
struct Person {
	string m_Name;  //C语言字符串数据类型，记得引用
	int m_Age;		 
};
void test03(){
	const Person p1 = {};		//const对象必须初始化.
	// 不可以直接改动，需要用指针
	// 指针可以改说明有分配地址
	Person* p = (Person*)&p1;  //获取指针
	p->m_Name = "XXXX";
	(*p).m_Age = 18;

	cout << "姓名：" << p1.m_Name << ", 年龄：" << p1.m_Age << endl;
}
int main() {
	test01();
	//test03();
	system("pause");
	return EXIT_SUCCESS;
}