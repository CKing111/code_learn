#include<iostream>
#include<string>
using namespace std;


// �����ԱΪ��ʱ������˳��Ϊ���Ƚ������һһ���죬Ȼ���ٹ��쵱ǰ��
// ����˳��Ϊ����������ǰ����������������ĳ�Ա��
// �ȹ���ĺ�������

class Phone {
public:
	Phone() { cout << "�ֻ���Ĭ�Ϲ��캯������!" << endl; }
	Phone(string name) { m_PhoneName = name; cout << "�ֵĵ��вι��캯�����ã�" << endl; }
	~Phone() { cout << "�ֻ���������������!" << endl; }

	string m_PhoneName;
};

class Game {
public:
	Game() { cout << "Game��Ĭ�Ϲ��캯�����ã�" << endl; }
	Game(string name) { m_GameName = name; cout << "Game���вι��캯�����ã�" << endl; }
	~Game() { cout << "Game�������������ã�" << endl; }

	string m_GameName;
};

class Person {
public:
	Person() { cout << "Person��Ĭ�Ϲ��캯������!" << endl; }
	Person(string name, string phoneName, string gameName):m_Name(name),m_Phone(phoneName),m_Game(gameName)
	{ 
		//m_Name = name; 
		cout << "Person���вι��캯�����ã�" << endl; 
	}

	void playGame() {
		cout << m_Name << "���š�" << m_Phone.m_PhoneName << "�����ֻ������š�" << m_Game.m_GameName << "����Ϸ��" << endl;
	}
	~Person() { cout << "Person�������������ã�" << endl; }

	string m_Name;
	Phone m_Phone;
	Game m_Game;
};


void test01() {
	Person p;
	p.m_Phone.m_PhoneName = "����";
	p.m_Game.m_GameName = "������";
}
void test02() {
	Person p("����", "����", "������");
	p.playGame();
}

int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}