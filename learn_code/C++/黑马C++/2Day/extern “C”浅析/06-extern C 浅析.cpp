#include<iostream>
using namespace std;

// extern "C"�������������C++�е���C���Է���������

// 1.������������
// ��Ҫʹ��extern C��ע�͵������ļ���
//#include"test.h"
//extern "C" void show();	// show()��C���Է�ʽ����

// 2.�����������
// ��C����ͷ�ļ���ͨ��ifdef���������ļ���
#include"test.h"


int main() {
	// show();		//1 ���޷��������ⲿ��� ��������
		// ԭ�򣺺����ڵ���ʱ�ᷢ�����أ�C������û�к������أ�����C++�༭��ʹ�ú����������޷�Ѱ�ҵ�����
	
	// ʹ�÷���2
	show2();
	show3();
	system("pause");
	return EXIT_SUCCESS;
}