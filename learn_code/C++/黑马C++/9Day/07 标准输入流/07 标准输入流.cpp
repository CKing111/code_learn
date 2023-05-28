#include<iostream>

using namespace std;

/*
cin.get() //一次只能读取一个字符
cin.get(一个参数) //读一个字符
cin.get(两个参数) //可以读字符串
cin.getline()
cin.ignore()
cin.peek()
cin.putback()
*/

void test01() {
	char c = cin.get();
	cout << "c = " << c << endl;

	c = cin.get();
	cout << "c = " << c << endl;

	c = cin.get();
	cout << "c = " << c << endl;

	c = cin.get();
	cout << "c = " << c << endl;
}
void test02() {
	// 输入两个参数可以读取字符串
	char buf[1024];
	cin.get(buf, 1024);		// 获取为buf字符串

	char c = cin.get();		// 检查'\n'是否被buf带走
	cout << buf << endl;

	if (c == '\n') {
		cout << "换行还在缓冲区" << endl;
	}
	else {
		cout << "换行不在缓冲区" << endl;
	}
}
// cin.get()读取字符串时不会将换行字符拿走
int main() {
	test02();
	system("pause");
	return EXIT_SUCCESS;
}