#include<iostream>

using namespace std;
// �̳з�ʽ��
// �����̳У�class ���� ��public ����{}--------���ɷ��ʸ���˽�У���������
// �����̳У�class ���� ��protected ����{}-----���ɷ��ʸ���˽�У������䱣��
// ˽�м̳У�class ���� ��private ����{}-------���ɷ��ʸ���˽�У�������˽��

// �����е�˽��Ҳ���̳У�ֻ�Ǳ����أ�����ͨ�����������鿴��cl /d1 reportSingleClassLayout+���� �ļ�����

// ����
class Base {
public:
	int m_A;
protected:
	int m_B;
private:
	int m_C;
};

class Son : public Base {
public:
	int m_D;
};

void test01() {
	cout << sizeof(Son) << endl;		// ���16����������Ӹ���һ���ĸ�int�����۱�������˽�ж������С
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}