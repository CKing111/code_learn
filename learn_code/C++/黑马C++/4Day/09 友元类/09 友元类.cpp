#include <iostream>
#include<string>
using namespace std;

// 可以先声明类，防止报错，然后后面再构建
class Building;

// 类
class GoodGay {
public:
	GoodGay();
	void visit();
private:
	Building* building;
};

class Building {
	// 告诉编译器，GoodGay是building的友元类，可以访问私有变量
	// 友元类是单项的，只能GoodGay访问building的
	// 不具有传递性
	friend class GoodGay;
public:
	Building();

	string m_SittingRoom;
private:
	string m_BedRoom;
};


Building::Building() {
	this->m_SittingRoom = "客厅";
	this->m_BedRoom = "卧室";

}
GoodGay::GoodGay()
{
	this->building = new Building;
}

void GoodGay::visit()
{
	cout << "好基友gg在访问：" << this->building->m_SittingRoom << endl;
	cout << "好基友gg在访问：" << this->building->m_BedRoom << endl;
}

void test01() {
	GoodGay gg;
	gg.visit();
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}

