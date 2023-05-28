#include<iostream>

using namespace std;

// C++ �쳣����

// �׳��Զ����쳣
class MyException {
public:
	void printError() {
		cout << "�Զ����쳣���Ͳ���" << endl; 
	}
};

// ջ����
class Person {
public:
	Person() {
		cout << "���캯��" << endl;
	}
	~Person() {
		cout << "��������" << endl;
	}
};

int myDivide(int a, int b) {
	// C ���Դ�����
	if (b == 0) {
		//return -1;

		// C++ �� �쳣���׳���ֻ��ע�׳��쳣����
		//throw 1;
		//throw 3.14;
		//throw "a";

		// ջ�������������쳣ʱ���ӽ��� try ��󣬵��쳣���׳�ǰ�����ڼ���ջ�Ϲ�������ж��󶼻ᱻ�Զ�������
		//			������˳���빹���˳���෴1 2����һ���̿��Ա�֤�쳣��ȫ�ԣ������ڴ�й©����Դռ��2��
		Person p1;
		Person p2;
		cout << "---------" << endl;
		/*
			���캯��
			���캯��
			---------
			��������
			��������
			�Զ����쳣���Ͳ���
		*/
		throw MyException();		// �׳�һ��MyException��������
	}
	return a / b;
}

void test01() {
	int a = 10;
	int b = 0;
	// C���Դ���ʽ
	myDivide(a, b);

	// C++ �����쳣
	try {		// ����ĳ�����쳣�������ú��������׳��쳣
		int ret = myDivide(a, b);
		cout << "ret �Ľ��Ϊ��" << ret << endl;
	}
	catch (int ) {		// �������׳����쳣��int����
		cout << "int ���͵��쳣������" << endl;
	}
	catch (double) {		// �������׳����쳣��double����
		// ���������쳣�󣬲����ٴ˴����쳣�������׳�
		throw;
		cout << "double ���͵��쳣������" << endl;
	}
	catch (MyException e) {
		e.printError();
	}
	catch (...) {		// �������׳����쳣��������������
		cout << "���� ���͵��쳣������" << endl;
	}
}



int main() {
	// main���������쳣
	try {
		test01();
	}
		catch (MyException e) {
		e.printError();
	}
	catch (...) {		// ��������񣬳�����Զ�����terminate�����ս����	
		//std::exception_ptr eptr = std::current_exception();
		//if (eptr) {
		//	try {
		//		std::rethrow_exception(eptr);
		//	}
		//	catch (const std::exception& e) {
		//		cout << "main����������" << typeid(e).name() << " ���͵��쳣" << endl;
		//		//throw;
		//	} 
		//	//catch (...) {
		//	//	cout << "main����������δ֪ ���͵��쳣" << endl;
		//	//}
		//}
		cout << "main�������������� ���͵��쳣" << endl;

	}
	system("pause");
	return EXIT_SUCCESS;
}