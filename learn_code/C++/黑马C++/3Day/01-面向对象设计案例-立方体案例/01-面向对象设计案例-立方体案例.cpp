#include<iostream>

using namespace std;

/*
	��Ŀ����
	�����������(Cube)���������������( 2*a*b + 2*a*c + 2*b*c )�����( a * b * c)��
	�ֱ���ȫ�ֺ����ͳ�Ա�����ж������������Ƿ���ȡ�
*/

// ��������
class Cube {
private:
	int m_L;	// ��
	int m_W;	// ��
	int m_H;	// ��

public:
	// ���ó�
	void setL(int l) { m_L = l;	}
	// ��ȡ��
	int getL()const { return m_L;	}  // ��Ա�����������޸�
	// ���ÿ�
	void setW(int w) { m_W = w; }
	// ��ȡ��
	int getW() { return m_W; }
	// ���ø�
	void setH(int h) { m_H = h; }
	// ��ȡ��
	int getH() { return m_H; }
	// �����������
	void getCubeS() {
		cout << "���������Ϊ��" << 2 * m_L * m_W + 2 * m_W * m_H + 2 * m_L * m_H << endl;
	}
	// �����������
	void getCubeV() {
		cout << "����������Ϊ��" << m_L * m_W * m_H << endl;
	}

	// 2.��Ա�����ж�
	bool compareCubeByClass(Cube &c){
		// �ó�Ա�����ж�
		bool ret = m_L == c.getL() &  m_W == c.getW() && m_H == c.getH();
		return ret;
	}
};



// �ж������������Ƿ���ȣ�����߶����,bool���ͺ���
// ���ô��룬ֱ����ҵ���ݱ��壬����Ҫ��ʱ���
// �������const�����Ա�������ܵ��ñ�������˽��Ա��const�±�����Ա���ܸı�
// const���ÿ��Զ�ȡ���ǲ����Ա��޸����ö����κζ�const���ý��и�ֵ���ǲ��Ϸ��ģ�������ָ��const��������ã�����const�����ò�������ָ��const��������á�
// 1.ȫ�ֺ����ж�
bool compareCube( Cube &c1, Cube &c2) {
	if(c1.getL() == c2.getL() && c1.getW() == c2.getW() && c1.getH() == c2.getH()){
		return true;
	}
	return false;
}
// �����������취����Ա����Ҳ���const
void func(const Cube & cub){
	cub.getL();		// ������getL������Ա�����������޸�
}



void test01() {
	Cube c1;
	c1.setL(10);
	c1.setH(10);
	c1.setW(10);

	c1.getCubeS();//600
	c1.getCubeV();//1000

	// ͨ��ȫ�ֺ����ж������������Ƿ����
	Cube c2;
	c2.setL(10);
	c2.setH(10);
	c2.setW(10);
	// �ж�
	bool ret = compareCube(c1, c2);
	if (ret) {
		cout << "ȫ�ֺ�����c1 �� c2 ����ȵģ���" << endl;
	}
	else {
		cout << "ȫ�ֺ�����c1 �� c2 �ǲ���ȵģ���" << endl;
	}

	// ͨ����Ա�����ж������������Ƿ����
	bool ret2 = c1.compareCubeByClass(c2);
	if (ret2) {
		cout << "��Ա������c1 �� c2 ����ȵģ���" << endl;
	}
	else {
		cout << "��Ա������c1 �� c2 �ǲ���ȵģ���" << endl;
	}
}
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}