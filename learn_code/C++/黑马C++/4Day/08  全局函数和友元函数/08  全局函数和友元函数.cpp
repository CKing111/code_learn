#include <iostream>;

using namespace std;

// ������
class Building {
	// ��Ԫ����
	// ��һ��ȫ�ֺ�������Ϊ�������Ԫ���������Է���˽�����ݣ�
	friend void goodGay(Building& b);
public:
	// ���캯������ʼ������
	Building(){
		this->m_SittingRoom = "����";
		this->m_BedRoom = "����";
	}

	// 
	string m_SittingRoom;	//  ����������

private:
	string m_BedRoom;		// ���ң�˽��
};

// ȫ�ֺ���������Ϊ��Ԫ����
void goodGay(Building& b) {
	cout << "�û������ڷ��ʣ�" << b.m_SittingRoom << endl;

	cout << "�û������ڷ��ʣ�" << b.m_BedRoom<< endl; // ʧ�ܣ�������Ԫfriend��ɹ�

}

void test01(){
	Building b;
	goodGay(b);
}
int main() {
	test01();
	system( "pause" );
	return EXIT_SUCCESS;
}