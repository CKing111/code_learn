#define _CRT_SECURE_ND_WARNINGS
#include<iostream>
#include"game1.h"
#include"game2.h"
using namespace std;


// �������ͷ�ļ�������ͬ�ĺ��������޷�ʶ��ᱨ��
// ���Խ�ͷ�ļ��еĺ�������һ�������ռ䣬ʹ��::����

// namespace�����ռ���Ҫ�������������ͻ������
// 1.�����ռ��¿��Էź������������ṹ�塢��
// 2.�����ռ����Ҫ������ȫ����������
// 3.�����ռ����Ƕ�������ռ�
// 4.�����ռ��ǿ��ŵģ�������ʱ��ԭ�ռ�������ݣ�ͬ���ռ��ϲ�
// 5.�������ռ䡢�����ռ�
// 6.�����ռ���������

namespace A {
	void Func();
	int m_A = 20;
	struct Person {

	};
	class Animal {};
	namespace B {
		int m_A = 10;
	}
}
namespace A {
	int m_B = 1999;
}
void test01() {
	LOL::goAtk();
	WZ::goAtk();
}
void test02() {
	cout << "������B�µ�m_AΪ��" << A::B::m_A<<endl;
}
void test03() {
	cout << "������A�µ�A::m_A:" << A::m_A << ", m_B:" << A::m_B << endl;
}

//�����ռ䣬�൱��������������̬����
namespace {
	int m_C = 100;  // static int m_C, static int m_D
	int m_D = 200;
}

// ����
namespace veryLongName {
	int m_E = 0;
}
void test04() {
	// ����
	namespace veryShortName = veryLongName;
	cout << veryLongName::m_E << endl;
	cout << veryShortName::m_E << endl;
}
int main() {
	test01();
	test02();
	test03();
	test04();
	system("pause");
	return EXIT_SUCCESS;
}