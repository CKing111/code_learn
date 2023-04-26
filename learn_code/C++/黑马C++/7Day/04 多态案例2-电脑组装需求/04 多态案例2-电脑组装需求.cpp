#include<iostream>

using namespace std;

// ������
class CPU {
public:
	// ����CPU���㷽��
	virtual void calculate() = 0;
};
// �����Կ�
class VideoCard {
public:
	virtual void display() = 0;
};
// �����ڴ�
class Memory {
public:
	virtual void storage() = 0;
};

// ������װ��
class computer{
public:
	// ����
	computer(CPU* cpu, VideoCard* card, Memory* mem) {
		this->cpu = cpu;
		this->card = card;
		this->memory = mem;
	}

	// ���Թ������������ø����๦�ܣ�����ʵ��
	void doWork() {
		this->cpu->calculate();	
		this->card->display();
		this->memory->storage();
	}

	// ����
	// ֻ���������������������Ϊ�麯��ʱ�����ܱ�֤��ʹ�� delete ���ٶ�̬����������ʱ��
	// ������������������ಢ�Ҿ����麯������ô�������������������Ҳ�ᱻ�Զ����á�������̳�Ϊ��̬������polymorphic destruction����
	virtual ~computer() {

		if (this->cpu != NULL) {
			delete this->cpu;
			this->cpu = NULL;
			cout << "����CPU" << endl;
		}
		if (this->card != NULL) {
			delete this->card;
			this->card = NULL;
			cout << "�����Կ�" << endl;
		}
		if (this->memory != NULL) {
			delete this->memory;
			this->memory = NULL;
			cout << "�����ڴ�" << endl;
		}
	}
	CPU* cpu;
	VideoCard* card;
	Memory* memory;
};

// ʵ�ֲ�
// intelCPU
class intelCPU :public CPU {
public:
	// CPU����
	virtual void calculate() {
		cout << "intelCPU��ʼ���㣡" << endl;
	}
};
// intel�Կ�
class intelVideoCard :public VideoCard {
public:
	virtual void display() {
		cout << "intel���Կ���ʼ��ʾ��" << endl;
	}
};
// intel�ڴ�
class intelMemory :public Memory {
public:
	virtual void storage() {
		cout << "intel�ڴ湤����" << endl;
	}
};

// lenovoCPU
class lenovoCPU :public CPU {
public:
	// CPU����
	virtual void calculate() {
		cout << "lenovoCPU��ʼ���㣡" << endl;
	}
};
// lenovo�Կ�
class lenovoVideoCard :public VideoCard {
public:
	virtual void display() {
		cout << "lenovo���Կ���ʼ��ʾ��" << endl;
	}
};
// lenovo�ڴ�
class lenovoMemory :public Memory {
public:
	virtual void storage() {
		cout << "lenovo�ڴ湤����" << endl;
	}
};

// ��װ���Բ���
void test01() {
	// ��һ̨����
	// ѡȡ���
	cout << "----��װ��һ̨intel����--------" << endl;
	CPU* cpu = new intelCPU;
	VideoCard* card = new intelVideoCard;
	Memory* mem = new intelMemory;

	// ��װ����
	computer* computer1 = new computer(cpu, card, mem);
	// ���е���
	computer1->doWork();

	/*
	��һ�������������ָ���Ա������Ҫ����������������ͷ����ָ����������ڴ�ռ䡣
	��ʹ�� delete �� delete[] ���ٶ�̬����������ʱ����Ҫ�����������������Ϊ��������������
	�����Զ�������������������Ӷ��ͷ����ָ����������ڴ�ռ䡣
	*/
	delete computer1;

	cout << "-----------------------------" << endl;
	cout << "----��װ�ڶ�̨lenovo����--------" << endl;
	CPU* cpu2 = new lenovoCPU;
	VideoCard* card2 = new lenovoVideoCard;
	Memory* mem2 = new lenovoMemory;

	// ��װ����
	computer* computer2 = new computer(cpu2, card2, mem2);
	// ���е���
	computer2->doWork();

	delete computer2;

	cout << "-----------------------------" << endl;
	cout << "----��װ����̨��ϵ���--------" << endl;
	CPU* cpu3 = new lenovoCPU;
	VideoCard* card3 = new intelVideoCard;
	Memory* mem3 = new intelMemory;

	// ��װ����
	computer* computer3 = new computer(cpu3, card3, mem3);
	// ���е���
	computer3->doWork();

	delete computer3;
}
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}