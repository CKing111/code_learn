//#include"Person.h"
//#include"Person.cpp"
#include"Person.hpp"

// 根据模板的特性，需要读取模板函数的实现才能成功构建类模板，因此需要直接引用cpp文件
// 如果将cpp实现与h头文件模板内容合并，则变为“.hpp”文件，通常是来保存模板代码


void test01() {
	Person<string, int> p("Tom", 29);
	p.showPerson();
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}