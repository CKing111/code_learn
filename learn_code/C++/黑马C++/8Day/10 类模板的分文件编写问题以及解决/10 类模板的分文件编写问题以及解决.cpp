//#include"Person.h"
//#include"Person.cpp"
#include"Person.hpp"

// ����ģ������ԣ���Ҫ��ȡģ�庯����ʵ�ֲ��ܳɹ�������ģ�壬�����Ҫֱ������cpp�ļ�
// �����cppʵ����hͷ�ļ�ģ�����ݺϲ������Ϊ��.hpp���ļ���ͨ����������ģ�����


void test01() {
	Person<string, int> p("Tom", 29);
	p.showPerson();
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}