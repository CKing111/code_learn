#include<iostream>

using namespace std;

/*
cin.get() //һ��ֻ�ܶ�ȡһ���ַ�
cin.get(һ������) //��һ���ַ�
cin.get(��������) //���Զ��ַ���
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
	// ���������������Զ�ȡ�ַ���
	char buf[1024];
	cin.get(buf, 1024);		// ��ȡΪbuf�ַ���

	char c = cin.get();		// ���'\n'�Ƿ�buf����
	cout << buf << endl;

	if (c == '\n') {
		cout << "���л��ڻ�����" << endl;
	}
	else {
		cout << "���в��ڻ�����" << endl;
	}
}
// cin.get()��ȡ�ַ���ʱ���Ὣ�����ַ�����
int main() {
	test02();
	system("pause");
	return EXIT_SUCCESS;
}