#include <iostream>
#include<string>
using namespace std;

// �����������࣬��ֹ����Ȼ������ٹ���
class Building;

// ��
class GoodGay {
public:
	GoodGay();
	void visit();
private:
	Building* building;
};

class Building {
	// ���߱�������GoodGay��building����Ԫ�࣬���Է���˽�б���
	// ��Ԫ���ǵ���ģ�ֻ��GoodGay����building��
	// �����д�����
	friend class GoodGay;
public:
	Building();

	string m_SittingRoom;
private:
	string m_BedRoom;
};


Building::Building() {
	this->m_SittingRoom = "����";
	this->m_BedRoom = "����";

}
GoodGay::GoodGay()
{
	this->building = new Building;
}

void GoodGay::visit()
{
	cout << "�û���gg�ڷ��ʣ�" << this->building->m_SittingRoom << endl;
	cout << "�û���gg�ڷ��ʣ�" << this->building->m_BedRoom << endl;
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

