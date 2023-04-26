#include <iostream>

using namespace std;

// int��ֵ����
void mySwapInt(int& a, int& b) {
	int tem = a;
	a = b;
	b = tem;
	
}
// double��ֵ����
void mySwapDouble(double & a, double & b) {
	double tem = a;
	a = b;
	b = tem;
}
// ��������������������


// ���ú���ģ��ʵ��ͨ�ú�������
template<typename T>    // ��ʾ����һ������ģ�壬����������Խ����κ����͵Ĳ�����
// typename �ؼ��ֱ�ʾ��ʵ���������Ͳ�����
// T ��������Ͳ��������ơ�T �Ϳ�������κ����ͣ���������ͽ��ڵ��ø�ģ��ʱ�ɱ������ƶϻ����ɳ���Ա�ֶ�ָ����
void mySwap(T& a, T& b) {		// ��������ֱ���޸�ԭʼ���ݣ���ͬ���Ͳ������㣬�������ò�ͬ���Ϳ�������
	T tem = a;
	a = b;
	b = tem;
}

// һ��ģ��ָ����һ���պ���������ʹ�ã��޷��Ƶ�T����
// ����ָ��ģ��T�����Ͳſ�ʹ��
template<typename T>
void mySwap2() {};

// ����ģ��ʹ��Ҫ��
// 1. �Զ������Ƶ��������ñ������Ƶ���һ�µ�T�ſ���ʹ��ģ��
//		eg��mySwap�� a, x )  // ʧ�ܣ�xΪchar��aΪint�������Ƶ���һ��T����
// 2. ��ʾָ������
//		eg: mySwap<int>(a,b)	// ��ʾָ�����Ͳ���TΪint����
//		eg: mySwap<double>(c,d)	// ��ʾָ�����Ͳ���TΪdouble����
void test01() {
	int a = 10;
	int b = 20;

	mySwapInt(a, b);

	cout << "int a = " << a << endl;
	cout << "int b = " << b << endl;

	double c = 10.1;
	double d = 20.1;

	mySwapDouble(c, d);

	cout << "double c = " << c << endl;
	cout << "double d = " << d << endl;

	cout << "-------------ʹ��template<typename T>����ģ���----------��" << endl;
	int e = 10;
	int f = 20;
	double g = 10.1;
	double h = 20.1;
	mySwap(e, f);
	mySwap(g, h);
	cout << "T e = " << e << endl;
	cout << "T f = " << f << endl;
	cout << "T g = " << g << endl;
	cout << "T h = " << h << endl;

	mySwap2<double>();		// �ɹ���ָ��T����
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}