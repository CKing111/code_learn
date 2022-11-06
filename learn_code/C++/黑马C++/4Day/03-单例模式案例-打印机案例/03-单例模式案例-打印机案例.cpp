#include<iostream>
#include<string>
using namespace std;

/*
	打印机案例：
		保护构造和拷贝构造；
		维护对象指针；
		声明公共的调用指针方法；
		类外实例化静态对象；
*/

class Printer {
public:
	static Printer* getInstance() {		// 静态成员函数，返回指针
		return singlePrinter;
	}
	// 功能
	void printText(string text){
		cout << text << endl;
		m_Count++;
		cout << "打印机使用次数：" <<m_Count<<endl;
	}
private:
	Printer() { m_Count = 0; }		// 用私有空间构造函数初始化私有变量
	Printer(const Printer& p) {}
	static Printer* singlePrinter;	// 单例对象，指针
	int m_Count;
};

Printer* Printer::singlePrinter = new Printer;	// 类外实例化

void test01() {
	// 拿到打印机对象
	Printer* printer = Printer::getInstance();
	
	// 调用方法
	printer->printText("离职报告");
	printer->printText("入职报告");
	printer->printText("入党报告");
	printer->printText("退休报告");
	printer->printText("请假报告");
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}