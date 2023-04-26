#define _CRT_SECURE_NO_WARNINGS // 添加宏定义，禁用编译器警告

#include<iostream>
#include<string>

#include<cstring>
using namespace std;


// 当存在堆区内容时，在正常构建子类父类析构函数时，发现子类析构函数无法正常调用，导致内存泄漏
// 解决办法，将析构变为虚析构，添加virtual关键字

// 纯虚析构
// 不同与纯虚函数，纯虚析构既要有声明也要有实现，类外实现
// 原因是base父类中也有可能有堆区内容，也需要析构函数释放内存，因此哪怕是纯虚函数也要有实现
// 只有纯虚析构函数也时抽象类，无法实例化对象
class Animal {
public:
	Animal() {
		cout << "调用Animal构造函数" << endl;
	};

	// 虚析构，子类有堆区内容时，可能会释放不干净，导致内存泄漏
	//virtual ~Animal() {
	//	cout << "调用Animal的析构函数" << endl;
	//}

	// 纯虚析构
	virtual ~Animal() = 0;

	// 虚函数
	virtual void speak() {
		cout << "动物在说话！" << endl;
	}
};

// 纯虚析构实现
Animal::~Animal() {
	cout << "Animal的纯虚析构调用" << endl;
}

class Cat :public Animal {
public:
	// 构造函数
	Cat(const char* name) {
		cout << "调用Cat构造函数" << endl;

		this->m_Name = new char[strlen(name) + 1];
		strcpy(this->m_Name, name);
	}

	virtual ~Cat() override {
		cout << "调用Cat的析构函数" << endl;
		if (this->m_Name != NULL) {
			delete [] this->m_Name;
			this->m_Name = NULL;
		}
	}

	virtual void speak() override {
		cout << "小猫在说话！" << endl;
	}

	char* m_Name; // Cat name
};

void test01() {
	Animal* animal = new Cat("Tom");
	animal->speak();

	delete animal;
}



int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}