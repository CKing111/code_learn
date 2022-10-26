#include<iostream>
#include<string>
using namespace std;

/*
	项目介绍：
		2.设计一个学生类，属性有姓名和学号，可以给姓名和学号赋值，可以显示学生的姓名和学号
	项目思路：
		
*/

class Student {
public:
// 成员属性
	string m_Name;
	int m_Id;

// 成员函数
	// 设置姓名
	void setName(string name) {
		m_Name = name;
	}
	// 设置学号
	void setId(int id) {
		m_Id = id;
	}
	// 打印信息
	void showInfo() {
		cout <<"学生姓名为： " << m_Name << ", 学生学号为： " << m_Id << endl;
	}
};

void test01() {
	// 创建一个学生，实例化--通过一个类来创建对象的过程
	Student s1;
	s1.setName("张三");
	s1.setId(1);

	// 通过s1属性打印信息
	cout << "s1的姓名为： " << s1.m_Name << ", s1的学号： " << s1.m_Id << endl;
	
	// 通过成员函数打印s1信息
	s1.showInfo();
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}