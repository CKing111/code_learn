#include<iostream>
using namespace std;

// ��������
// 1. ֵ���ݣ����ı�ԭʼֵ
void mySwap(int a , int b) {
	int tmp = a;
	a = b;
	b = tmp;

	cout << "mySwap::a = " << a << endl;
	cout << "mySwap::b = " << b << endl;
}
void test01() {
	int a = 10;
	int b = 20;
	mySwap(a, b);  //ֵ����

	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
}

// 2.��ַ����
void mySwap2(int* a, int* b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
	cout << "mySwap2::a = " << *a << endl;
	cout << "mySwap2::b = " << *b << endl;
}
void test02() {
	int a = 10;
	int b = 20;
	mySwap2(&a, &b);	// ��ַ����
	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
}


// 3.���ô��ݣ����Ƶ�ַ����
// ���ݱ����ķ�ʽ����ԭʼֵ���β�=ʵ��
// ���þ���ָ�볣������
void mySwap3(int& a, int& b) {		//&a = a
	int tmp = a;
	a = b;
	b = tmp;
}
void test03() {
	int a = 10;
	int b = 20;
	mySwap3(a, b);	//���ô���
	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
}


// ���õ�ע������
// 1.���ñ�����һ��Ϸ����ڴ�ռ䣻
// 2.��Ҫ���ؾֲ����������ã�
int doWork() {
	int a = 10;
	return a;
}
void test04() {
	// int &a = 10;

	//int& ret = doWork();	// �����ԣ��ֲ���������
	//cout << "ret = " << ret << endl;
	//cout << "ret = " << ret << endl;
	//cout << "ret = " << ret << endl;
	//cout << "ret = " << ret << endl;
}

// 3.��������ֵ�����ã���ô�������������Ϊ��ʽ��ֵ
int& doWork2() {
	static int a = 10; // ��ǰ�ļ���ȫ�ֱ���
	return a;
}
void test05() {
	// int &a = 10;
	int&ret = doWork2();	// ���ԣ���������ֵ����
	cout << "ret = " << ret << endl;
	cout << "ret = " << ret << endl;
	cout << "ret = " << ret << endl;
	cout << "ret = " << ret << endl;
	doWork2() = 1000; //�ȼ���a=1000
	cout << "a = " << doWork2() << endl;

}
//int main() {
	//test01();
	//test02();
	//test03();
	//test04();
	//test05();
	//system("pause");
	//return EXIT_SUCCESS;
//}