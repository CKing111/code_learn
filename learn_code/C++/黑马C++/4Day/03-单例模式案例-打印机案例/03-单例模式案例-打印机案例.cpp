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
		cout <<"打印： "<< text << endl;
		m_Count++;
		cout << "打印机使用次数：" <<m_Count<<endl;
	}
private:
	Printer() { m_Count = 0; }		// 用私有空间构造函数初始化私有变量
	Printer(const Printer& p) {}	// 拷贝构造
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

/*
1.构造函数和拷贝构造函数都被设为私有函数，
外部无法直接创建对象或拷贝对象，从而保护单例对象不被重复创建或复制。

2.维护一个静态的 Printer 类型指针 singlePrinter，用于指向单例对象。

3.通过一个公共的静态成员函数 getInstance() 来获取该单例对象，
该方法返回的是单例对象的指针，如果单例对象不存在，则在方法内部进行创建。

4.类外实例化单例对象，也就是在 Printer 类外部通过静态指针 
singlePrinter 来创建 Printer 类的单例对象。

5.在使用单例对象时，通过 Printer::getInstance() 方法来获取该对象的指针，
从而保证多次调用该方法得到的是同一个对象，即单例对象。
*/