#include<iostream>

using namespace std;
/*
虚继承的原理是通过在继承体系中建立虚基类表来实现。当一个类使用虚继承时，
编译器会为该类生成一个虚基类表（virtual base table），用来记录该类所继承的虚基类的位置和偏移量等信息。

当派生类从虚基类继承时，它会通过虚基类表来访问虚基类的成员，这样就可以避免多次继承同一个虚基类而导致的二义性问题。
在派生类中，对于虚基类的成员的访问也是通过虚基类表来实现的。

虚基类表本质上是一个指针数组，其中每个元素都指向对应的虚基类在对象内存布局中的位置。
在派生类中，虚基类的偏移量也会被记录在虚基类表中，这样就可以通过虚基类表来计算虚基类成员的地址。

虚继承的实现需要编译器在编译时进行一些额外的处理，因此会增加一定的开销。
但是，在多重继承的情况下，虚继承可以避免二义性问题，提高代码的可维护性和可读性，因此被广泛使用。
*/
// 动物类
class Animal {
public:

	int m_Age;
};

// 羊类
// 虚继承
class Sheep:virtual public Animal {
public:

	//int m_Age;
};

// 鸵类
// 虚继承
class Tuo :virtual public Animal {
public:

	//int m_Age;
};

// 羊驼类
class SheepTuo :public Sheep, public Tuo {

};

void test01() {
	SheepTuo st;
	// 如何解决继承的二义性

	// 方法一：作用域，但会造成一侧数据的浪费
	st.Sheep::m_Age = 10;			
	st.Tuo::m_Age = 20;
	/*
class SheepTuo  size(8):
		+---
 0      | +--- (base class Sheep)
 0      | | +--- (base class Animal)
 0      | | | m_Age
		| | +---
		| +---
 4      | +--- (base class Tuo)
 4      | | +--- (base class Animal)
 4      | | | m_Age
		| | +---
		| +---
		+---
*/


	// 方法二：虚继承，定义子类继承时添加virtual，此时父类变为虚基类
	cout << "st.m_Age = " << st.m_Age << endl;
	cout << "st.Sheep::m_Age = " << st.Sheep::m_Age << endl;
	cout << "st.Tuo::m_Age = " << st.Tuo::m_Age << endl;
	/*class SheepTuo  size(20):
        +---
 0      | +--- (base class Sheep)
 0      | | {vbptr}
        | | <alignment member> (size=4)
        | +---
 8      | +--- (base class Tuo)
 8      | | {vbptr}
        | | <alignment member> (size=4)
        | +---
        +---
        +--- (virtual base Animal)
16      | m_Age
        +---
		虚基类表：
		SheepTuo::$vbtable@Sheep@:
		 0      | 0
		 1      | 16 (SheepTuod(Sheep+0)Animal)

		SheepTuo::$vbtable@Tuo@:
		 0      | 0
		 1      | 8 (SheepTuod(Tuo+0)Animal)
		vbi:       class  offset o.vbptr  o.vbte fVtorDisp
				  Animal      16       0       4 0
当一个类使用虚继承时，它可能从多个类中继承同一个虚基类，这就会导致虚基类在派生类中出现多次。
为了解决这种二义性问题，编译器需要为派生类生成一个虚基类表，用来记录派生类所继承的虚基类的位置和偏移量等信息。
这些信息包括：

虚基类的地址：记录虚基类在对象内存布局中的位置，这样可以在派生类中访问虚基类的成员。
虚基类在派生类中的偏移量：记录虚基类在派生类对象内存布局中的偏移量，这样就可以通过偏移量计算虚基类成员的地址。

虚基类表本质上是一个指针数组，其中每个元素都指向对应的虚基类在对象内存布局中的位置。
在派生类中，虚基类的偏移量也会被记录在虚基类表中，这样就可以通过虚基类表来计算虚基类成员的地址。

需要注意的是，虚基类表的生成和使用是由编译器来完成的，程序员一般无需手动干预。
虚基类表是 C++ 编译器为支持虚继承所实现的机制之一，通过使用虚基类表，编译器能够保证虚继承的正确性，
避免派生类中出现多个虚基类的实例而导致的二义性问题。
		*/
}

// 虚继承内部工作原理
void test02() {
	SheepTuo st;
	st.m_Age = 100;

	// 通过sheep找到偏移量地址
	// 获取地址并解引用到虚基类表
	cout << "通过sheep找到偏移量为：" << *(int*)((int*)*(int*)&st + 1) << endl;
	/*
	&st 取得对象st的地址，即SheepTuo对象的地址。
	*(int*)&st 将对象st的地址强制转换为指向虚基类指针的指针类型，并解引用该指针，获取虚基类指针的值。
	(int*)*(int*)&st 将虚基类指针的值强制转换为指向整型的指针类型，并解引用该指针，获取虚基类表的地址。
	*(int*)((int*)*(int*)&st + 1) 在虚基类表中找到偏移量的地址。由于虚基类表中的第一个元素是指向虚函数表的指针，
	因此偏移量的地址是在第二个元素的位置。因此在虚基类表的地址上加上1，就可以得到偏移量的地址。
	最后再解引用该地址，获取偏移量的值。
	将偏移量的值输出到控制台上。
*/
	cout << "通过Tuo找到的偏移量为：" << *(int*)((int*)*((int*)&st + 1) + 1) << endl;

	// 通过偏移量求出m_Age的值
	cout << "age = " << ((Animal*)((char*)&st + (*(int*)((int*)*(int*)&st + 1))))->m_Age << endl;
}


int main() {
	//test01();
	test02();

	system("pause");
	return EXIT_SUCCESS;
}

