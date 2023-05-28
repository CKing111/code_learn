#include<iostream>
#include<string>
using namespace std;

// Ŀ���ǻ���ϵͳ��׼�쳣��д�Լ�����Ҫ���쳣��Ϣ

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

// �Լ����쳣��
class MyOutOfRange :public exception {		// �����ڱ�׼�쳣�ĸ���
public:
	// ���������Ϣ��string��
	MyOutOfRange(char* errorInfo) {
		// ��char* ת��Ϊstring
		// ����һ��
		this->m_ErrorInfo = string(errorInfo);
	}
	// ����2									
	MyOutOfRange(string errorInfo) {
		// ��char* ת��Ϊstring
		// ����һ��
		this->m_ErrorInfo = errorInfo;
	}
	// ��д������麯����~exception()��what()
	virtual ~MyOutOfRange()
	{
	}
	 virtual char const* what() const		// �ڶ���const��������������thisָ��
	{
		// string תΪchar*
		 return this->m_ErrorInfo.c_str();	//  string ��� c_str() ��Ա����������ǰ�ַ���ת��Ϊ C ����ַ�����
	}

	 string m_ErrorInfo;
};

class Person {
public:
	Person(int age) {
		if (age < 0 || age>150) {
			// ��������Խ���쳣�׳�
			//throw out_of_range("�������Ҫ�� 0 �� 150 ֮�䣡��");
			//throw MyOutOfRange(string("�Զ����쳣���������Ҫ�� 0 �� 150 ֮�䣡��"));		// ����string
			throw MyOutOfRange("�Զ����쳣���������Ҫ�� 0 �� 150 ֮�䣡��");				// ����char
		}
		this->m_Age = age;
	}

	int m_Age;
};


void test01() {
	try {
		Person p1(1511);
	}
	catch (exception& e) {		// ֱ��ʹ���쳣��̬�ĸ���
		cout << e.what() << endl;	// ϵͳ��׼�쳣����һ��.what()�������쳣��Ϣ
	}
	//catch (MyOutOfRange& e) {		// 
	//	cout << e.what() << endl;	// ϵͳ��׼�쳣����һ��.what()�������쳣��Ϣ
	//}
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}