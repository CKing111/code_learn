#include<iostream>

#define _CRT_SECURE_NO_WARNINGS

using namespace std;


// C++中的封装，严格类型转换检测，属性和行为是封装在一起的
// 1.属性和行为作为一个整体来进行处理
// 2.控制权限：共有权限public,保护权限protected,私有权限private
//		class中默认是private权限
//		struct中默认权限是public权限
// 3.权限：
//		私有权限：就是私有成员（属性、函数）在类内可以访问，类外不可以访问，子类也不可以访问；
//		公共权限：在类内外都可以访问的；
//		保护权限：类内部可以访问，当前类的子类可以访问，类外部不可以访问
struct Person {
	char mName[64];
	int mAge;
	void PersonEat() {
		cout << mName << "吃人饭！" << endl;
	}
};
struct Dog {
	char mName[64];
	int mAge;
	void DogEat() {
		cout << mName << "吃狗饭！" << endl;
	}
};

void test01() {
	Person p1;
	strcpy_s(p1.mName, strlen("老王") + 1, "老王");
	p1.PersonEat();

	Dog d1;
	strcpy_s(d1.mName, strlen("旺财") + 1,"旺财");
	d1.DogEat();

	// p1.DogEat();		// 失败，不属于当前实例的封装函数
}

class Animal {
// 默认私有权限，可以类内访问
	void eat() { mAge = 100; mHight = 180; mWeight = 70; };	//类内访问
	int mAge;
// 公共权限
public:
	int mHight;
// 保护权限
protected:
	int mWeight;
};

void test02() {
	Animal a1;
	//a1.eat();	// 私有权限
	//a1.mAge = 100;	// 私有权限
	a1.mHight = 180;	// 公共权限
	//a1.mWeight = 70;	// 保护权限
}
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;

}