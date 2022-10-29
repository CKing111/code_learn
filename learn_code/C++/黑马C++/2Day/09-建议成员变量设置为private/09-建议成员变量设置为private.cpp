#include<iostream>
#include<string>

using namespace std;

// 封装class，将成员函数设置为private

class Person {
private:				// 类外不可访问，类内访问
	int m_Age = 0;		// 年龄，只读
	string m_Name;		// 公有权限，读写
	string m_Love;		// 情人，只写

public:
	// 设置年龄
	void setAge(int age) {
		if (age < 0 || age>100) {
			cout << "你这个老妖精！！" << endl;
			m_Age = age;
			return;		// 通过后return掉
		}
		m_Age = age;
	}
	// 读取年龄
	int getAge() {
		return m_Age;
	}
	// 读取姓名
	string getName() {
		return m_Name;
	}
	// 写姓名；
	void setName(string name) {
		m_Name = name;
	}
	// 写入情人
	void setLove(string lover) {
		m_Love = lover;
	}
};

void test01() {
	Person p1;
	//p1.m_name;		// 不可读
	p1.setName("老王"); // 写入姓名
	cout << "p1姓名：" << p1.getName() << endl;	//获取姓名
	cout << "p1的年龄：" << p1.getAge() << endl;//获取年龄
	p1.setAge(101);		// 设置年龄
	cout << "p1的年龄：" << p1.getAge() << endl;//获取年龄
	p1.setLove("张三");
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}