#include<iostream>
using namespace std;

// extern "C"就是用来解决：C++中调用C语言方法的问题

// 1.单个函数调用
// 需要使用extern C并注释掉引用文件名
//#include"test.h"
//extern "C" void show();	// show()用C语言方式链接

// 2.多个函数调用
// 在C语言头文件中通过ifdef处理，引用文件名
#include"test.h"


int main() {
	// show();		//1 个无法解析的外部命令， 链接问题
		// 原因：函数在调用时会发生重载，C语言中没有函数重载，所以C++编辑器使用函数重载名无法寻找到函数
	
	// 使用方法2
	show2();
	show3();
	system("pause");
	return EXIT_SUCCESS;
}