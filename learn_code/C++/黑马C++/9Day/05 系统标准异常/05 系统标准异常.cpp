#include<iostream>

using namespace std;

// ʹ��ϵͳ��׼�쳣��Ҫ����
#include<stdexcept>
/*
<stdexcept>�а������µ��쳣�ࣺ
	std::logic_error����ʾ�����߼�������쳣�ࡣ
	std::domain_error����ʾ����������Ч����쳣�ࡣ
	std::invalid_argument����ʾ��Ч�������쳣�ࡣ
	std::length_error����ʾ���ȳ������Ƶ��쳣�ࡣ
	std::out_of_range����ʾ���ʳ�����Χ���쳣�ࡣ
	std::runtime_error����ʾ����ʱ������쳣�ࡣ
	std::overflow_error����ʾ������������쳣�ࡣ
	std::underflow_error����ʾ������������쳣�ࡣ
*/

class Person {
public:
	Person(int age) {
		if (age < 0 || age>150) {
			// ��������Խ���쳣�׳�
			throw out_of_range("�������Ҫ�� 0 �� 150 ֮�䣡��");
		}
		this->m_Age = age;
	}

	int m_Age;
};

void test01() {
	try {
		Person p1(151);
	}
	//catch (out_of_range& e) {
	//	cout << e.what() << endl;
	//}
	catch (exception& e) {		// ֱ��ʹ���쳣��̬�ĸ���
		cout << e.what() << endl;	// ϵͳ��׼�쳣����һ��.what()�������쳣��Ϣ
	}
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}