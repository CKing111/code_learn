#include<iostream>

using namespace std;

int myPlus1(int a, int b) {
	return a + b;
}

template<typename T>
T myPlus2(T a, T b) {
	return a + b;
}


void test01() {
	int a = 10;
	int b = 20;
	char c = 'c';
	// ��ͨ������ģ�溯��������
	cout << "��ͨ�������Խ�����ʽ����ת����char->int: a+c = 10+'c' = 10 + 99 = " << myPlus1(a, c) << endl;
	cout << "myPlus2(a,c)ʧ�ܣ��Զ������Ƶ���ʽ�����Խ�����ʽ����ת��" << endl;
	cout << myPlus2<int>(a, c) << ", myPlus2<int>(a,c)��ʾָ�����ͷ�ʽ�ɹ���" << endl;
}

// ���ߵ��ù���
// 1. �����ͨ�����ͺ���ģ�����ͬʱ���ã�����ѡ����ͨ�������߼���
// 2. �����ǿ�Ƶ��ú���ģ���е����ݣ�����ʹ�ÿղ����б�
template<class T>
void myPrint(T a, T b) {
	cout << "����ģ��1����" << endl;;
}
template<class T>
void myPrint(T a, T b, T c) {
	cout << "����ģ��2����" << endl;
}
void myPrint(int a, int b) {
	cout << "��ͨ��������" << endl;
}

void test02(){
	int a = 0;
	int b = 0;
	int c = 0;
	myPrint(a, b);		// ������ͨ����
	myPrint<>(a, b);	// ǿ�Ƶ���ģ��
	myPrint(a, b, c);	// ����ģ������
	}
int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}