#include<iostream>

using namespace std;

/*
	����ģʽ������Ϊ�˴������еĶ��󣬲��ұ�ֻ֤��һ������ʵ�������ܹ��졢����һ���µ���ͬ��ʵ����
				����ģʽͨ�������Զ��ͷſռ䣬Ĭ��ֻ��һ��ָ��ռ�ռ�С
	���裺
		�����캯���Ϳ�������˽�л���
		�ڲ�ά��һ������ָ�룻
		˽�л�Ψһ�Ķ���ָ�룻
		�����ṩgetInstangce�����������������ָ�룻
		��֤����ֻ��ʵ����һ������

*/



// ������ϯ��
class ChairMan {

private:	
	// ˽�л������캯��
	ChairMan() {
		cout << "����˽�пռ�ChairMan���캯����" << endl;
	}
	// ��Ҫ�����������������private��
	static ChairMan* singeMan;		// ������̬����ά��һ����ָ�룬��������
	// ��������˽�л�
	ChairMan(const ChairMan& c) {}


public:		// ����ռ�
	// �ṩ��̬��Աget���������ʹ���������
	// �÷����ǵģ������޷�������ΪNULL
	static ChairMan* getInstance() {
		return singeMan;
	}
};
// �����ʼ��
ChairMan* ChairMan::singeMan = new ChairMan;

void test01() {
	//ChairMan c1;
	//ChairMan* c2 = new ChairMan;
	//ChairMan* c3 = new ChairMan;

	// ����˽�пռ���޷�ʹ��
	//ChairMan::singeMan;		// ��������
	//// ָ���ȡ
	//ChairMan* cm = ChairMan::singeMan;
	//ChairMan* cm2 = ChairMan::singeMan;

	// ʹ��get��������
	//ChairMan::getInstance() = NULL;			// ʧ�ܣ�get����û��Ȩ���޸�ָ��Դ
	ChairMan* cm1 = ChairMan::getInstance();	// �ɹ�����ȡָ��
	ChairMan* cm2 = ChairMan::getInstance();
	if (cm1 == cm2) {
		cout << "cm1 �� cm2 ��ͬ��" << endl;
	}
	else {
		cout << "cm1 �� cm2 ����ͬ��" << endl;
	}

	// ������Ĭ�Ͽ�������ʱ������������ͬһ���������ݣ�cm2 != cm3
	//ChairMan* cm3 = new ChairMan(*cm2);
	//if (cm2 == cm3) {
	//	cout << "cm3 �� cm2 ��ͬ��" << endl;
	//}
	//else {
	//	cout << "cm3 �� cm2 ����ͬ��" << endl;
	//}

}

int main() {
	cout << "Main�������ã�" << endl;

	test01();

	system("pause");
	return EXIT_SUCCESS;
}