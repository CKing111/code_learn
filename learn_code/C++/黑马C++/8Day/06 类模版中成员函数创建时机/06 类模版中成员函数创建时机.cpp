#include<iostream>

using namespace std;

class Person1 {
public:
	void showPerson1() {
		cout << "Person1 show" << endl;
	}
};

class Person2 {
public:
	void showPerson2() {
		cout << "Person2 show" << endl;
	}
};


// ��ģ���еĳ�Ա����������һ��ʼ�ʹ��������ģ�
// ���������н׶βŴ���������

template<class T>
class myClass {
public:
	// ֻ�����к�Ż�ֻ�������������ĳ�Ա����Ա����

	void func1() {
		obj.showPerson1();
	}

	void func2() {
		obj.showPerson2();
	}
	T obj;		
};

void test01() {
	myClass<Person1> p1;
	p1.func1();
	//p1.func2();		// ʧ�ܣ��Ƶ���T���Ͳ�֧��person2������
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}