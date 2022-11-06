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
		cout << text << endl;
		m_Count++;
		cout << "��ӡ��ʹ�ô�����" <<m_Count<<endl;
	}
private:
	Printer() { m_Count = 0; }		// ��˽�пռ乹�캯����ʼ��˽�б���
	Printer(const Printer& p) {}
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