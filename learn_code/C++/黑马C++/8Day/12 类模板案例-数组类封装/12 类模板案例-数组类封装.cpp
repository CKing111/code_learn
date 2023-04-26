#include<iostream>
#include<string>
#include"myArray.hpp"

using namespace std;

// 打印函数
void printIntArr(MyArray<int>& myArr) {
	for (int i = 0; i < myArr.getSize(); i++) {
		cout << myArr[i] << " ";
	}
	cout << endl;
}

// 测试int类型数组
void test01() {
	MyArray<int> myIntArr(100);
	for (int i = 0; i < 10; i++) {
		myIntArr.pushBack(i + 100);
	}
	// 打印数组
	printIntArr(myIntArr);
}

// 测试自定义数据类型
class Person {
public:
	Person() {};
	Person(string name, int age) :m_Name(name), m_Age(age) {}		// 列表初始化
	string m_Name;
	int m_Age;
};
void printPerson(MyArray<Person>& myArr) {
	for (int i = 0; i < myArr.getSize(); i++) {
		cout << "姓名：" << myArr[i].m_Name << ", 年龄：" << myArr[i].m_Age << endl;
	}
}
void test02() {
	// 测试自定义数据类型Person
	MyArray <Person> personArr(10);
	Person p1("孙悟空1",599);
	Person p2("孙悟空2",19);
	Person p3("孙悟空3",29);
	Person p4("孙悟空4",39);
	Person p5("孙悟空5",49);

	personArr.pushBack(p1);
	personArr.pushBack(p2);
	personArr.pushBack(p3);
	personArr.pushBack(p4);
	personArr.pushBack(p5);
	
	// 打印
	printPerson(personArr);
}

int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}