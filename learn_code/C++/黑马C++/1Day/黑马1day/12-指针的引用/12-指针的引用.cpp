#include<iostream>

using namespace std;

struct Person {
	int m_Age;
};

// 分配内存空间，本质是对指针进行内存分配作业
// 方式1：用二级指针修改指针指向
// p:具体的Person对象；*p:对象的指针；**p:指针的指针
void allocatMemory(Person** p) {	//导入二级指针
	*p = (Person*)malloc(sizeof(Person));  // 分配二级指针指向指针的位置
		/*
			在使用malloc开辟空间时，使用完成一定要释放空间，如果不释放会造内存泄漏。
			malloc函数返回的实际是一个无类型指针，必须在其前面加上指针类型强制转换才可以使用
			指针自身 = (指针类型*）malloc（sizeof（指针类型）*数据数量）
		*/
	(*p)->m_Age = 100;	// 取出指针*p所指向结构体的m_Age元素并赋值
}

void test01() {
	Person* p = NULL;	// 声明一个无效空指针，NULL是一个宏：#define NULL 0，
						/*
								因为内存从0开始的一段区域正常情况下是不允许读写的，
							所以我们规定，“当指针数值为0时，也就是它指向内存地址0时，
							这个指针就是不正常的指针，也就是我们所要声明的“该指针当前无效””。
							这样，指针就无法再进行任何数据访问了。
								编程工作中有一类比较容易犯的错误–指针地址未进行正确的更新赋值就加以使用，
							这往往会造成很严重的后果（对内存区进行错误的涂抹）。所以一个良好的习惯是，
							当一个指针的工作稍事休息，先把它赋值为NULL，待到再度使用时，
							重新对其赋值以及进行指针类型转化。
						*/
	cout <<"初始化p值： "<< p << endl;
	// 分配内存
	allocatMemory(&p);
	cout << "二级指针，p的年龄： " << p->m_Age << endl;
}

// 方式二利用指针引用开辟空间
// 直接操作引用生成的指针常量，不需要进行二级指针操作
// 相当于通过传入指针别名，对指针进行作业
void allocatMemoryByRef(Person* &p) {
	p = (Person*)malloc(sizeof(Person));  // 分配引用空间
	p->m_Age = 1000;	// 引用就是指针，直接赋值
}
void test02() {
	Person* p = NULL;	// 声明一个无效空指针，NULL是一个宏：#define NULL 0，
	// 分配内存
	allocatMemoryByRef(p); // 不需要传递地址，引用自己解析地址
	cout << "指针引用，p的年龄： " << p->m_Age << endl;
}

int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;

}