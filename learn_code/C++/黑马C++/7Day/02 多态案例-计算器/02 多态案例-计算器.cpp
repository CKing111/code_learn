#include<iostream>
#include<string>

using namespace std;

// ����ԭ�򣺿���ԭ��--����չ���п��ţ����޸Ľ��йر�
//				�ȣ���ֻ��Ҫ�ṩ����base�࣬���ܻ��ڴ˽�����չ�����޸�base
// ��̬�ĺô��������չ�ԣ���ǿ��֯�ԣ��ɶ���ǿ
//				��������������麯�������ಢû����д������麯�����ǽ���������
//				������಻��д������麯�����Ƕ�̬�������ã��һ����Ӵ����ڲ����ӳ̶�
// 
// ���ö�̬ʵ�ּ�����
// ������������
class AbstractCalculator {
public:
	// �����麯��������������չ
	//virtual int getResult() {
	//	return 0;
	//}

	// Ҳ�������ô��麯�������ǲ�����ʵ��������
	// �д��麯�������Ϊ�����࣬���޷�ʵ��������ģ�
	// �����������дʵ�ָ���Ĵ��麯������������Ҳ�ǳ����� 
	// ֻ���麯�����ǳ�����
	virtual int getResult() = 0;

	int m_A;
	int m_B;
};

// �ӷ�������
/*
class AddCalculator     size(16):
		+---
 0      | +--- (base class AbstractCalculator)
 0      | | {vfptr}
 8      | | m_A
12      | | m_B
		| +---
		+---

AddCalculator::$vftable@:
		| &AddCalculator_meta
		|  0
 0      | &AddCalculator::getResult
*/
class AddCalculator :public AbstractCalculator {
public:
	virtual int getResult() {
		return m_A + m_B;
	}
};

// ����������
class SubCalculator :public AbstractCalculator {
public:
	virtual int getResult() {
		return m_A - m_B;
	}
};

// �˷�������
class MultiCalculator :public AbstractCalculator {
public:
	virtual int getResult() {
		return m_A * m_B;
	}
};

// ����������
class DivisionCalculator :public AbstractCalculator {
public:
	virtual int getResult() {
		return m_A / m_B;
	}
};


void test01() {
	// ������ʹ�üӷ�������
	AbstractCalculator* calculator = new AddCalculator;
	calculator->m_A = 20;
	calculator->m_B = 10;
	cout <<"20+10 = " << calculator->getResult() << endl;

	// �ͷŸ�������������
	delete calculator;
	calculator = new SubCalculator;
	calculator->m_A = 20;
	calculator->m_B = 10;
	cout << "20-10 = " << calculator->getResult() << endl;
}



int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;

}