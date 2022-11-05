#include<iostream>
#include<string>
using namespace std;


// 当类成员为类时，构造顺序为，先将类对象一一构造，然后再构造当前类
// 析构顺序为，先析构当前类对象，在析构类对象的成员类
// 先构造的后析构！

class Phone {
public:
	Phone() { cout << "手机的默认构造函数调用!" << endl; }
	Phone(string name) { m_PhoneName = name; cout << "手的的有参构造函数调用！" << endl; }
	~Phone() { cout << "手机的析构函数调用!" << endl; }

	string m_PhoneName;
};

class Game {
public:
	Game() { cout << "Game的默认构造函数调用！" << endl; }
	Game(string name) { m_GameName = name; cout << "Game的有参构造函数调用！" << endl; }
	~Game() { cout << "Game的析构函数调用！" << endl; }

	string m_GameName;
};

class Person {
public:
	Person() { cout << "Person的默认构造函数调用!" << endl; }
	Person(string name, string phoneName, string gameName):m_Name(name),m_Phone(phoneName),m_Game(gameName)
	{ 
		//m_Name = name; 
		cout << "Person的有参构造函数调用！" << endl; 
	}

	void playGame() {
		cout << m_Name << "拿着《" << m_Phone.m_PhoneName << "》牌手机，玩着《" << m_Game.m_GameName << "》游戏！" << endl;
	}
	~Person() { cout << "Person的析构函数调用！" << endl; }

	string m_Name;
	Phone m_Phone;
	Game m_Game;
};


void test01() {
	Person p;
	p.m_Phone.m_PhoneName = "三星";
	p.m_Game.m_GameName = "斗地主";
}
void test02() {
	Person p("老王", "三星", "斗地主");
	p.playGame();
}

int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}