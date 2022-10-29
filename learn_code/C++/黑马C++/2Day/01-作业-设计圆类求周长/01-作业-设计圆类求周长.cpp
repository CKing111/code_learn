#include<iostream>

using namespace std;

/*
	项目介绍：
		1.	设计一个类，求圆的周长。
	项目思路：
		通过面向对象思路，设计一个生成对象圆的类来计算周长
		周长公式： 2 * pi * r
*/
// 1.声明pi常量,用const替代#define
const double pi = 3.14;
// 2.设计一个求圆周长的类
class Circle {
public: // 公共权限
// 声明属性
	int m_R;	// 半径,成员属性

// 声明成员函数
	// 求周长成员方法
	double calculateZC() {
		return 2 * pi * m_R;
	}

	// 设置半径的成员方法
	// 成员函数是可以修改公共成员属性
	void setR(int r) {
		m_R = r;
	}
};

// 
void test01() {
	// 通过类创建一个圆
	Circle c1;  // 圆（对象）
	// 1.直接成员赋值方法
	//c1.m_R = 10;
	// 2.通过成员函数，间接给圆设计半径
	c1.setR(10);
	// 打印c1圆周长输出
	cout << "c1的周长： " << c1.calculateZC() << endl;
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}