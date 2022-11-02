#include<iostream>
using namespace std;


class Person {
public:  // 构造和析构函数必须放在public下才可以调用
	// 普通构造函数
	Person() { cout << "默认构造函数调用！" << endl; }		// 默认构造函数（无参）
	Person(int a) {
		m_Age = a;			// 参数构造函数可传参
		cout << "有参构造函数调用！" << endl;
	}	// 有参构造函数
	// 拷贝构造函数, 固定格式
	// 作用就是赋值类的内容
	// 拷贝函数必须加const，不允许拷贝过程修改内容
	// 必须加&，因为不加为值传递，会生成根据拷贝的Person类型的 p 对象生成临时对象，临时对象还是会调用拷贝构造，然后会变成死循环
	Person(const Person& p) {
		m_Age = p.m_Age;		// 赋值拷贝对象的公共参数
		cout << "拷贝构造函数调用！" << endl;
	}

	~Person() { cout << "析构函数调用！" << endl; }			// 析构函数

	int m_Age;		// 公共参数
};

// 拷贝构造使用场景，使用时机
// 1.用已经创建好的对象来初始化新的对象
void test01() {
	Person p1;
	p1.m_Age = 100;

	Person p2(p1);
	cout << "p2的年龄：" << p2.m_Age << endl;
}

// 2. 以值传递的方式给函数参数传值，值传递不会对原始数据进行修改
//  函数值传递时就是通过拷贝构造函数传递进去值，引用传递不会调用拷贝构造函数
//  值传递会生成中间的临时值，这个值使用拷贝构造生成
void doWork(Person p) {		// 函数传参表示为：拷贝构造Person p = Person(p1)，调用了拷贝构造
	cout << "函数值传递Person p 的年龄：" << p.m_Age << endl;
}
void test02() {
	Person p1;
	p1.m_Age = 100;

	doWork(p1);
}

// 3.以值的方式返回局部对象
//  不能以引用返回局部对象，会改变值
Person doWork2() {		// 返回值
	Person p1;	// 默认构造函数
	p1.m_Age = 100;
	return p1;	// 拷贝构造函数
}
void test03() {
	Person p = doWork2();	// 使用函数返回值构造Person p对象
	cout << "拷贝构造生成函数返回值p的年龄：" << p.m_Age << endl;
}

// vs的release模式：自动又换代码，能调用引用就不用拷贝
// test03优化为：
/*
	Person p;	// 不调用默认构造函数
	doWork2(p);
	void doWork2(Person &p){
		Person p1;		// 调用默认构造函数
	}
*/
int main() {
	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}