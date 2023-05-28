#include<iostream>;

using namespace std;

// �쳣����
class BaseException {
public:
	virtual void printError() = 0;		// ���麯��
};


// ��ָ���쳣
class NULLPointException:public BaseException{
	virtual void printError() {
		cout << "��ָ���쳣" << endl;
	}
};

// Խ���쳣
class OutOfRangeException :public BaseException {
	virtual void printError() {
		cout << "Խ���쳣" << endl;
	}
};

void doWork() {
	//throw NULLPointException();
	throw OutOfRangeException();
}

void test01() {
	try {
		doWork();
	}
	catch(BaseException & e){
		e.printError();
	}
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}