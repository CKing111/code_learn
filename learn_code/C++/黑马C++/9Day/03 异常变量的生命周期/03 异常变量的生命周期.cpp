#include<iostream>

using namespace std;

class MyException {
public:
	MyException() {
		cout << "MyException�Ĺ��캯������" << endl;
	}
	MyException(const MyException& e) {
		cout << "MyException�Ŀ������캯������" << endl;
	}
	~MyException() {
		cout << "MyException��������������" << endl;
	}
};

void doWork() {
	//throw MyException();
	//throw & MyException();	// ����MyException *eʱ���ã��Զ��ͷ�
	throw new MyException();	// ����������MyException *eʱ�������Զ��ͷ�

}

void test01() {
	try {
		doWork();
	} 
	//catch (MyException &e) {		// �������õķ�ʽ������ÿ������죬�������Ч��
	//	cout << "MyException���쳣����" << endl;
	//}
	catch (MyException* e) {		
		cout << "MyException���쳣����" << endl;
		delete e;
	}
	// MyException e:����ÿ�������
	// MyException &e:���ã�������ÿ������죬����
	// MyException *e:ָ��,ֱ���׳���������&MyException()���ͷŵ��������ٲ�������e
	// MyException *e:ָ��,ֱ���׳���������new MyException()���ͷŵ������ٲ�������e����Ҫ�ֶ��ͷ�e��������Ч����ͬ

}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}