#include<iostream>

using namespace std;

// 1.ȫ�ֱ�����ǿ
//
//int a;
int a = 30;

// 2.���������ǿ������������ǿ�����غ�����ǿ
// ����ָ���β����ͣ�ʹ��ʱ����������Ӧ
int getRectS(int w, int h) {
	return w * h;
};
void test01() {
	cout << getRectS(10, 10)<<endl;
};

// 3.���ͼ��ת����ǿ
// ��ͬ���ͼ��ת��������������ǿ��ת��
void test02() {
	// char* p = malloc(sizeof(64)); //malloc����ָ�룬����ֵΪvoid*
	char* p = (char*)malloc(sizeof(64));
}

// 4.struc��ǿ
struct Person {
	int m_Age;
	void plusAge() { m_Age++; };	// C++��struct�п������Ӻ���
};
void test03() {
	Person p1; // C++���Բ�����struct
	p1.m_Age = 10;
	p1.plusAge();
	cout << p1.m_Age << endl;
}

// 5.bool������ǿ C������û��bool����
bool flag = true;	// ֻ�����٣�true�����棨��0����false����٣�0��
void test04() {
	cout << sizeof(bool) << endl;
	cout << flag << endl;
	flag = 100;  // Ĭ��ת����1
	cout << flag << endl;
}

// 6.��Ŀ�������ǿ��C++���ص��Ǳ���
void test05() {
	int a = 30;
	int b = 40;

	cout << "ret = " << (a > b ? a : b) << endl;
	(a > b ? a : b) = 100; //ͨ����b=100��C���Է��ص��Ǳ���
	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
}

// 7.const ��ǿ
const int m_A = 10; // ȫ�ֳ������ܵ������������
void test06() {
	//m_A = 100;
	const int m_B = 20; //C++���������ĳ��������ᷢ���ı�
	//m_B = 100;

	int* p = (int*)&m_B;
	*p = 200;
	/*
		C++�еȼ��ڿ�����һ����ʱ�ڴ�ռ�洢��������ʽΪ��
		int tmp = m_B;
		int *p = (int *)&tmp;
		*pָ����ʱ�ռ䣬ԭʼ����m_Bû�иı䣻	
	*/
	cout << "*p = " << *p << ", m_B = " << m_B << endl;
}

int main() {
	//test01();
	//test03();
	//test04();
	//test05();
	test06();
	system("pause");
	return EXIT_SUCCESS;
}