#include<iostream>

using namespace std;

// �꺯��ȱ��
// 1. ����ӷ�
#define MyAdd(x,y) x+y
#define MyAdd2(x,y) (x+y)
#define MAX 1024
// ����1������δ�ϸ�����������
void test01() {
	int ret = MyAdd(10, 20);
	cout << "define: ret = " << ret << endl;
	int ret2 = MyAdd(10, 20) * 20;  // Ԥ�ڽ������10+20��*20=600
									// �������� 10+20*20 = 410
	cout << "define: ret2 = " << ret2 << endl;
	int ret3 = MyAdd2(10, 20) * 20;  // Ԥ�ڽ������10+20��*20=600
	cout << "define: ret3 = " << ret3 << endl;
}

// ����2����Ԫ�ػ��ظ�����
#define MyCompare(a,b) ((a)<(b)) ? (a):(b)
void test02() {
	int a = 10;
	int b = 20;

	int ret = MyCompare(a, b); // Ԥ�������10
	cout << "define: ret = " << ret << endl;
	// ����Ԫ��
	int ret2 = MyCompare(++a, b);// Ԥ����11��������12
								 // ����++a��<(b)��? (++a:b) 
								 // Ԫ���ڱȽϺ����ʱ����ִ����++����
	cout << "define: ret2 = " << ret2 << endl;
}

// ����3���꺯��û��������

// �������������ú�������ʽ����������������ֵ�����
inline int myadd(int a, int b) { return a + b; }
inline int mycompare(int a, int b) { return a < b ? a : b; }
void test03() {
	int a = 10;
	int b = 20;

	int ret = myadd(a, b);
	int ret2 = mycompare(a, b);
	int ret3 = mycompare(++a, b);
	cout << "myadd, ret = " << ret << endl;
	cout << "mycompare(a,b),ret2 = " << ret2 << endl;
	cout << "mycompare(++a,b),ret3 = " << ret3 << endl;
}

// 1.��������ע������
//		��������������ʵ�ֶ���Ҫ�ӹؼ���inline
//		���ڲ��ĳ�Ա������Ĭ�ϱ�Ϊ��������
inline void func() {};

int main() {
	//test01();
	//test02();
	test03();
	cout << MAX << endl;
	system("pause");
	return EXIT_SUCCESS;
}