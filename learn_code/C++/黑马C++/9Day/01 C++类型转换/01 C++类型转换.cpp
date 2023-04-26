#include<iostream>

using namespace std;

// 1. 静态类型转换，
void test01() {

	// 内置数据类型
	char a = 'a';
	
	// static_cast<目标类型> (原对象)
	double d = static_cast<double>(a);

	cout << d << endl;
}

// 自定义类型
class Base {
	virtual void func() {};
};
class Son :public Base{
	virtual void func() {};		// 多态条件，重写父类虚函数
};
class Other {};
void test02() {
	// 自定义数据类型
	Base* base = NULL;
	Son* son = NULL;

	// base转变为Son类型  --- 向下类型转换，不安全
	Son* son2 = static_cast<Son*>(base);
	// son转为Base* 类型------向上类型转换，安全
	Base* base2 = static_cast<Son*>(son);
	cout << son2 << endl;

	// base 转换为Other*
	// 失败：没有父子关系的两个类型之间是无法转换成功的
	//Other* oth = static_cast<Base*>(base);
}


// 2. 动态类型转换
void test03() {
	// 内置数据类型
	// 失败：动态类型转换不允许内置数据类型之间的转换
	//char c = 'c';
	//double d = dynamic_cast<double>(c);

	// 自定义数据类型
	Base* base = NULL;
	Son* son = NULL;

	// base 转化为Son*类型，不安全
	// 失败，不安全转换不成功，只有编程多态才会转变成
	//Son* son2 = dynamic_cast<Son*>(base);

	// son 转为 Base* , 安全, 成功
	Base* base2 = dynamic_cast<Base*>(son);

	// base 转换为 Other*，失败，非父子无关系，无法多态
	//Other* oth = dynamic_cast<Other*>(base);

	// 如果发生多态，那么父子之间的转换总是安全的
	Base* base3 = new Son;		// 多态使用方式，父类指针或者引用指向子类对象
	// 向下类型转换不安全,basez转向Son*
	Son* son3 = dynamic_cast<Son*>(base3);

	/*
	这句话的意思是，如果你有一个父类指针或引用，它可能指向一个父类对象，也可能指向一个子类对象。如果你想将它转换为子类指针或引用，你必须确定它实际上是指向一个子类对象的，否则你会得到一个错误的类型转换。例如，如果你有一个 Animal 类和一个 Dog 类，Dog 是 Animal 的子类，那么你可以这样写：

	Animal* a = new Dog(); // a 指向一个 Dog 对象，可以向下类型转换
	Dog* d = dynamic_cast<Dog*>(a); // d 也指向同一个 Dog 对象，类型转换成功

	Animal* b = new Animal(); // b 指向一个 Animal 对象，不能向下类型转换
	Dog* e = dynamic_cast<Dog*>(b); // e 为 nullptr，类型转换失败
	*/
}

// 3. 常量转换
// const_cast<type-id>(expression)
// 不可以对非指针和非引用做const_cast
void test04() {
	// 指针之间的转换
	const int* p = NULL;
	// 将const int * 转换为 int *
	int* p2 = const_cast<int*>(p);

	// 将p2 转换为 const int *类型
	const int* p3 = const_cast<const int*>(p2);

	// 引用之间的转换
	const int a = 10;
	const int& aRef = a;	// 取 a的地址

	// Const int & 转换为 int &
	int& aRef2 = const_cast<int&>(aRef);
	
	// 非指针非引用转换失败
	//int b = const_cast<int>(a);
	
	/*
	常量转换有以下用途：

	可以在常量成员函数中修改非常量成员变量2。
	可以将指向常量对象的指针或引用转换为指向非常量对象的指针或引用，从而可以修改原对象的值1 4。
	但这种做法只有在原对象本身不是常量时才合法，否则会导致未定义行为4。
	可以将指向易变对象的指针或引用转换为指向非易变对象的指针或引用，从而可以忽略原对象的易变性3。
	*/
}

// 4. 重新解释转换，最不安全，不建议用,reinterpret_cast<type-id>(expression)
// 可以将一个指针或引用的类型转换为与其完全不同的类型，通常是不兼容的类型。
void test05() {
	// int -> int*
	int a = 10;
	int* p = reinterpret_cast<int*>(a);

	// Base * -> Other * ;
	Base* base = NULL;
	Other* other = reinterpret_cast<Other*>(base);

	/*
	有以下用途：

	可以将一个指针转换为一个足够大的整数类型，或者将一个整数类型转换为一个指针3。
	可以将一个指针转换为一个与之无关的类的指针，或者将一个成员指针转换为一个与之无关的类的成员指针1 2。
	可以在一些特殊的场合下，如哈希函数1，位操作4，或者与硬件相关的编程中，使用重新解释转换来改变指针或引用的底层二进制表示4。
	*/
}
int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}