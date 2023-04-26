#include<iostream>

using namespace std;
// 继承方式：
// 公共继承：class 子类 ：public 父类{}--------不可访问父类私有，其他不变
// 保护继承：class 子类 ：protected 父类{}-----不可访问父类私有，其他变保护
// 私有继承：class 子类 ：private 父类{}-------不可访问父类私有，其他变私有

// 父类中的私有也被继承，只是被隐藏，可以通过其他方法查看（cl /d1 reportSingleClassLayout+类名 文件名）

// 父类
class Base {
public:
	int m_A;
protected:
	int m_B;
private:
	int m_C;
};

class Son : public Base {
public:
	int m_D;
};

void test01() {
	cout << sizeof(Son) << endl;		// 输出16，代表子类加父类一共四个int，无论保护还是私有都会算大小
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}