#include<stdio.h>
#include<string.h>
#include<stdlib.h>

//using namespace std;

// ȫ�ֱ�����ǿ
// 
int a;
int a = 30;

// ���������ǿ������������ǿ�����غ�����ǿ
int getRectS(w, h) {
	//return w * h;
};
// 3.���ͼ��ת����ǿ
void test02() {
	char* p = malloc(sizeof(64)); //malloc����ָ�룬����ֵΪvoid*

}

// 4.struc��ǿ
struct Person {
	int m_Age;
	//void plusAge();	// C����struct�в��������Ӻ���
};
void test03() {
	struct Person p1; // C���Կ��Բ�����struct
	//p1.m_Age = 10;
	//p1.plusAge();
	//std::cout << p1.m_Age << std::endl;
}

// 5.bool������ǿ C������û��bool����
// bool flag;
// 
// 6.��Ŀ�������ǿ��C���Է��ص���ֵ
void test05() {
	int a = 30;
	int b = 40;
	printf("ret = %d \n", a > b ? a : b);
	// a > b ? a : b = 100; //����20=100��C���Է��ص���ֵ
}

// 7.const ��ǿ
const int m_A = 10; //�ܵ��������ɸ���
void test06() {
	//m_A = 100;
	const int m_B = 20; //C�����У�����ͨ��ָ��ı�ֵ����α����
	//m_B = 100;

	int* p = (int*)&m_B;
	*p = 200;
	printf("*p = %d, m_B = %d", *p, m_B);
}
int main() {
	//test01();
	//test02();
	//test05();
	test06();
	system("pause");
	return EXIT_SUCCESS;
}