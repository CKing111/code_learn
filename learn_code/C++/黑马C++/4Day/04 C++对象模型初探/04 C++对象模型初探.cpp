#include <iostream>;

using namespace std;

#pragma pack(show) // ����ģ��

// ֪ʶ��1: ���С�ļ���
// ֪ʶ��2�� thisָ��
class Person {
public:
	int m_A;		// ��С4�� ��Ա���ԣ�����Person��Ĵ�С��
	double m_C;		// ��С8
	void func() {		// Ĭ��thisָ��
		m_A = 100;
	}; // ��Ա������������Ĵ�С�У�

	static int m_B; // ��̬��Ա������Ҳ���������С��

	static void func2() {};	// ��̬��Ա����Ҳ�������С
};

int Person::m_B = 0;


void test01() {
	cout << sizeof(Person) << endl;		// ��personΪ��ʱ��ռ�ڴ�Ϊ1
	// ����Ҳ�ǿ���ת��Ϊʵ���ģ����Լ��ĵ�ַ
	// Person p[10] : &p[1] != &p[0]

	// thisָ��ָ�򱻵��õĳ�Ա���������Ķ���
	Person p1;
	p1.func();	// func( this -> p1)

	Person p2;
	p2.func();	// func( this -> p2)
}
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}
