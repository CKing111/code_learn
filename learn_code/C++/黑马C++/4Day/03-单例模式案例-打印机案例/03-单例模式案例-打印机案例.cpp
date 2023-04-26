#include<iostream>
#include<string>
using namespace std;

/*
	��ӡ��������
		��������Ϳ������죻
		ά������ָ�룻
		���������ĵ���ָ�뷽����
		����ʵ������̬����
*/

class Printer {
public:
	static Printer* getInstance() {		// ��̬��Ա����������ָ��
		return singlePrinter;
	}
	// ����
	void printText(string text){
		cout <<"��ӡ�� "<< text << endl;
		m_Count++;
		cout << "��ӡ��ʹ�ô�����" <<m_Count<<endl;
	}
private:
	Printer() { m_Count = 0; }		// ��˽�пռ乹�캯����ʼ��˽�б���
	Printer(const Printer& p) {}	// ��������
	static Printer* singlePrinter;	// ��������ָ��
	int m_Count;
};

Printer* Printer::singlePrinter = new Printer;	// ����ʵ����

void test01() {
	// �õ���ӡ������
	Printer* printer = Printer::getInstance();
	
	// ���÷���
	printer->printText("��ְ����");
	printer->printText("��ְ����");
	printer->printText("�뵳����");
	printer->printText("���ݱ���");
	printer->printText("��ٱ���");
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}

/*
1.���캯���Ϳ������캯��������Ϊ˽�к�����
�ⲿ�޷�ֱ�Ӵ�������򿽱����󣬴Ӷ������������󲻱��ظ��������ơ�

2.ά��һ����̬�� Printer ����ָ�� singlePrinter������ָ��������

3.ͨ��һ�������ľ�̬��Ա���� getInstance() ����ȡ�õ�������
�÷������ص��ǵ��������ָ�룬����������󲻴��ڣ����ڷ����ڲ����д�����

4.����ʵ������������Ҳ������ Printer ���ⲿͨ����ָ̬�� 
singlePrinter ������ Printer ��ĵ�������

5.��ʹ�õ�������ʱ��ͨ�� Printer::getInstance() ��������ȡ�ö����ָ�룬
�Ӷ���֤��ε��ø÷����õ�����ͬһ�����󣬼���������
*/