#include<iostream>
#include"MyString.h"

void test01() {
	MyString str("aaa");
	cout << str << endl;		// 重载<<
	cout << "请重新输入str!!" << endl;

	cin >> str;					// 重载>>
	cout << "重新输入后，str为："<<str << endl;

	MyString str2(str);
	cout << "拷贝构造的str2为：" << str2 << endl;

	// 输出索引值
	cout << "str[1]为：" << str[1] << endl;		// 重载[]

	str[1] = 'n';
	cout << "修改str[1]后，为：" << str << endl;		// 重载[]

	// 带指针的等号赋值
	MyString str3 = " ";
	str3 = str;						// 等号重载
	cout << "重载赋值运算符后，str3为：" << str3 << endl;		// 重载[]

	// 重载加法运算符
	MyString str4 = "abc";
	MyString str5 = "def";
	MyString str6 = str4 + str5;		// 返回值

	cout << "重载加法运算符后，str6为：" << str6 << endl;		// 重载[]

	// 重载==
	if (str == str3) {
		cout << "str 和 str3相等！" <<str<<", "<<str3 << endl;
	}
	else {
		cout << "str 和 str3 不相等！" << str << ", " << str3 << endl;
	}

}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}