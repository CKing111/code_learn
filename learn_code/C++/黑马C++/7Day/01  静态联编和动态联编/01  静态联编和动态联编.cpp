#include<iostream>

using namespace std;

class Animal {
public:
	// 增加virtual关键字，speak变虚函数，
	virtual void speak() {
		cout << "动物在说话！" << endl;
	}

	virtual void eat(int a) {
		cout << "动物在吃饭！" << endl;
	}
};

class Cat :public Animal {
public:
	// 重写虚函数
	void speak() {
		cout << "猫在说话！" << endl;
	}
	void eat(int a) {
		cout << "小猫在吃饭！" << endl;
	}
};

class Dog :public Animal {
public:
	void speak() {
		cout << "狗在说话！" << endl;
	}
	void eat(int a) {
		cout << "小狗在吃饭！" << endl;
	}
};

// 1. 对于有继承关系的类，c++可以不用通过类型强转
// 2. 静态联编---地址已经绑定，已经限定父类成员函数输出，输入子类还是会调用父类
// 3. 动态联编---地址没有绑定死，父类成员函数变虚函数，增加virtual关键字

// 4. 多态的满足条件：
// 1）. 父类有虚函数（virtual）
// 2）. 子类必须重写该虚函数（重写是指函数声明完全一样）
// 3）. 父类的指针或者引用指向子类的对象，eg：Animal & animal = Cat & cat
// 4）. 子类重写过程中，可以不加virtual


/*
虚函数多态是指通过基类指针或引用指向派生类对象，并调用虚函数时，实际运行的是派生类的实现。
这种多态性可以用于在基类中定义通用的接口和行为，而由派生类自定义实现细节，从而实现代码的灵活扩展和维护。

实现虚函数多态需要在基类函数的声明前加上 virtual 关键字，
这样编译器在进行函数调用时会根据实际对象类型选择正确的函数实现，而不是根据指针或引用类型选择函数实现。

class Cat       size(8):
		+---
 0      | +--- (base class Animal)
 0      | | {vfptr}
		| +---
		+---

Cat::$vftable@:
		| &Cat_meta
		|  0
 0      | &Cat::speak
*/
void doSpeak(Animal & animal) {
	animal.speak();
}
void doEat(Animal& animal) {
	animal.eat(10);
}

void test01() {
	Cat cat;
	Dog dog;

	doSpeak(cat);	// 成功，Animal & animal = Cat & cat，return： Animal
	doSpeak(dog);	// 成功，Animal & animal = Cat & cat，return： Animal

	doEat(cat);
	doEat(dog);
}

void test02() {
	Animal* animal = new Cat;		// 多态，父类指向子类
	// 底层实现，等价于：animal->speak();
	// *(int *)*(int *)animal   表示Cat::speak函数在表中的地址
	((void(*)())(*(int*)*(int*)animal))();
	/*
	解析：
		1）第一步：*(int *)*(int *)animal，通过指针运算精确定位Cat类speak函数在派生类虚表中的地址
			根据C++的对象模型，对于有虚函数的类，它的对象中存储一个指向该类虚表的指针，也称为虚指针。
			虚表是将该类中所有虚函数的地址按照定义顺序存放在一个表中，虚指针指向该表的起始地址。
			在派生类中重新定义父类的虚函数时，会在虚表中插入其新的实现地址，覆盖父类中的原地址，从而实现多态性。
			因此，通过解引用animal指针和两次转换将其转换为一个int类型指针，即可得到Cat类对象虚指针指向的虚表地址。
			然后，再次解引用该地址，得到Cat类对象在虚表中speak函数的入口地址
		2）第二步：将Cat类speak函数的入口地址转换为无参无返回值的函数指针：(void(*)())...
			由于该指针需要用于最后的函数调用，因此要将Cat类speak函数的入口地址转换为一个指针类型，
			该指针类型表示一个无参无返回值的函数。这可以通过C++中的指针类型强制转换语法实现。
			最后，将Cat类speak函数的入口地址转换为无参无返回值的函数指针后，使用函数调用语法将其执行，
			即可实现对Cat类speak成员函数的调用。
	*/

	// 实现animal->eat();
	/*
	class Cat       size(8):
        +---
 0      | +--- (base class Animal)
 0      | | {vfptr}
        | +---
        +---

Cat::$vftable@:
        | &Cat_meta
        |  0
 0      | &Cat::speak
 1      | &Cat::eat

	*/
	//((void(*)())* ((int*)*(int*)animal + 1))();

	// C++默认调用惯例是cdecl，现改用__stdcall
	typedef void(__stdcall* FUNC)(int);
	(FUNC(*((int*)*(int*)animal + 1)))(10);
}

int main() {
	test01();

	test02();

	cout << sizeof(Animal) << endl;
	system("pause");
	return EXIT_SUCCESS;


}