#include<iostream>

using namespace std;

namespace C {
	int s = 20;
}
namespace D {
	int s = 30;
}
// using ������using ***::***;
// using ����ָ�using namespace ***��
// ��ȡ�����оͽ�ԭ�򣬵��������ռ�;ֲ�����ͬ��ʱ������ֶ����ԣ�
// using�����������ռ䣬���������õĻ������ǻ��ھͽ�ԭ��
// ��������ռ�������ͬ�����������ָ��Ҳ������쳣��

void test01() {
	//int s = 30;
	// using C::s; // ������
	cout << D::s << endl;
}

void test02() {
	int s = 30;
	// using ����ָ��
	using namespace C;
	cout << C::s << endl;
	cout << D::s << endl;
	cout << s << endl;
}

int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}

