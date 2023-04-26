#include<iostream>

using namespace std;

// 抽象类
class CPU {
public:
	// 抽象CPU计算方法
	virtual void calculate() = 0;
};
// 抽象显卡
class VideoCard {
public:
	virtual void display() = 0;
};
// 抽象内存
class Memory {
public:
	virtual void storage() = 0;
};

// 电脑组装类
class computer{
public:
	// 构造
	computer(CPU* cpu, VideoCard* card, Memory* mem) {
		this->cpu = cpu;
		this->card = card;
		this->memory = mem;
	}

	// 电脑工作函数，调用各自类功能，抽象实现
	void doWork() {
		this->cpu->calculate();	
		this->card->display();
		this->memory->storage();
	}

	// 析构
	// 只有在类的析构函数被声明为虚函数时，才能保证在使用 delete 销毁动态分配的类对象时，
	// 如果该类派生自其他类并且具有虚函数，那么它的派生类的析构函数也会被自动调用。这个过程称为多态析构（polymorphic destruction）。
	virtual ~computer() {

		if (this->cpu != NULL) {
			delete this->cpu;
			this->cpu = NULL;
			cout << "析构CPU" << endl;
		}
		if (this->card != NULL) {
			delete this->card;
			this->card = NULL;
			cout << "析构显卡" << endl;
		}
		if (this->memory != NULL) {
			delete this->memory;
			this->memory = NULL;
			cout << "析构内存" << endl;
		}
	}
	CPU* cpu;
	VideoCard* card;
	Memory* memory;
};

// 实现层
// intelCPU
class intelCPU :public CPU {
public:
	// CPU计算
	virtual void calculate() {
		cout << "intelCPU开始计算！" << endl;
	}
};
// intel显卡
class intelVideoCard :public VideoCard {
public:
	virtual void display() {
		cout << "intel的显卡开始显示！" << endl;
	}
};
// intel内存
class intelMemory :public Memory {
public:
	virtual void storage() {
		cout << "intel内存工作！" << endl;
	}
};

// lenovoCPU
class lenovoCPU :public CPU {
public:
	// CPU计算
	virtual void calculate() {
		cout << "lenovoCPU开始计算！" << endl;
	}
};
// lenovo显卡
class lenovoVideoCard :public VideoCard {
public:
	virtual void display() {
		cout << "lenovo的显卡开始显示！" << endl;
	}
};
// lenovo内存
class lenovoMemory :public Memory {
public:
	virtual void storage() {
		cout << "lenovo内存工作！" << endl;
	}
};

// 组装电脑测试
void test01() {
	// 第一台电脑
	// 选取配件
	cout << "----组装第一台intel电脑--------" << endl;
	CPU* cpu = new intelCPU;
	VideoCard* card = new intelVideoCard;
	Memory* mem = new intelMemory;

	// 组装电脑
	computer* computer1 = new computer(cpu, card, mem);
	// 运行电脑
	computer1->doWork();

	/*
	在一个类中如果包含指针成员，则需要在类的析构函数中释放这个指针所分配的内存空间。
	当使用 delete 或 delete[] 销毁动态分配的类对象时（需要将类的析构函数声明为虚析构函数），
	它会自动调用类的析构函数，从而释放这个指针所分配的内存空间。
	*/
	delete computer1;

	cout << "-----------------------------" << endl;
	cout << "----组装第二台lenovo电脑--------" << endl;
	CPU* cpu2 = new lenovoCPU;
	VideoCard* card2 = new lenovoVideoCard;
	Memory* mem2 = new lenovoMemory;

	// 组装电脑
	computer* computer2 = new computer(cpu2, card2, mem2);
	// 运行电脑
	computer2->doWork();

	delete computer2;

	cout << "-----------------------------" << endl;
	cout << "----组装第三台混合电脑--------" << endl;
	CPU* cpu3 = new lenovoCPU;
	VideoCard* card3 = new intelVideoCard;
	Memory* mem3 = new intelMemory;

	// 组装电脑
	computer* computer3 = new computer(cpu3, card3, mem3);
	// 运行电脑
	computer3->doWork();

	delete computer3;
}
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}