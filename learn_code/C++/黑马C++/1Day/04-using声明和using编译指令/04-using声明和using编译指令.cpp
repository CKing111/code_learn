#include<iostream>

using namespace std;

namespace C {
	int s = 20;
}
namespace D {
	int s = 30;
}
// using 声明：using ***::***;
// using 编译指令：using namespace ***；
// 读取变量有就近原则，调用命名空间和局部变量同名时，会出现二义性；
// using编译会打开命名空间，但不必须用的话，还是基于就近原则；
// 多个命名空间中有相同变量，如果不指定也会出现异常；

void test01() {
	//int s = 30;
	// using C::s; // 二义性
	cout << D::s << endl;
}

void test02() {
	int s = 30;
	// using 编译指令
	using namespace C;
	cout << C::s << endl;
	cout << D::s << endl;
	cout << s << endl;
}

int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}

