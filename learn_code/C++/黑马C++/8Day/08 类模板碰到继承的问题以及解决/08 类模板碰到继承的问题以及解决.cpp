#include<iostream>

using namespace std;

// ���ڼ̳й�ϵʱ���и���Ϊ��ģ��
// ��ʱ�������ڴ���ʱ�򣬱����������ģ��T�����ͣ����ܷ��丸����ڴ�

template<class T>
class Base {
public:

	T m_A;
};

template<class T1, class T2>
class Son : public Base<T2> {
public:

	T1 m_B;
};


void test01(){
	Son<int, double> s;
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}