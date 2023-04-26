#include <iostream>;

using namespace std;


class MyFunc {
public:

	// 重载小括号（），当输入str时，直接输出
	void operator()(string text) {
		cout << text << endl;
	}
};

void test01() {

	// 仿函数，本质还是重载，但可以模仿函数输出
	MyFunc func;
	func("hello world!!");

}

class MyAdd {
public:
	int operator()(int a, int b) {
		return a + b;
	}
};

void test02(){
	MyAdd add;
	cout << add(1, 2) << endl;

	// 匿名对象函数
	// 直接使用重载（）
	cout << MyAdd()(10, 10) << endl;
}

int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}