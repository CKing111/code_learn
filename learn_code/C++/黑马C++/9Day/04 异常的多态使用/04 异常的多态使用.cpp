#include<iostream>;

using namespace std;

// 异常基类
class BaseException {
public:
	virtual void printError() = 0;		// 纯虚函数
};


// 空指针异常
class NULLPointException:public BaseException{
	virtual void printError() {
		cout << "空指针异常" << endl;
	}
};

// 越界异常
class OutOfRangeException :public BaseException {
	virtual void printError() {
		cout << "越界异常" << endl;
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