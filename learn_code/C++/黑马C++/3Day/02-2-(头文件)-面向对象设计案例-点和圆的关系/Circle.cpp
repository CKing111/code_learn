#include"Circle.h"

void Circle::setR(int r) { m_R = r; }	// ���ð뾶
int Circle::getR() { return m_R; }		// ��ȡ�뾶
void Circle::setCenter(Point p) { m_Center = p; }	// ����Բ��
Point Circle::getCenter() { return m_Center; }		// ��ȡԲ��
// 1.����Բ���Ա�����жϵ�Բ��ϵ
void Circle::isInCircleByClass(Point& p) {
	// ��ȡԲ�ĵ���ľ���ƽ��
	int Distance = ((m_Center.getX() - p.getX()) * (m_Center.getX() - p.getX())) + ((m_Center.getY() - p.getY()) * (m_Center.getY() - p.getY()));
	// ��ȡ�뾶ƽ��
	int rDistance = m_R * m_R;
	// �Ա�λ�ù�ϵ
	if (rDistance == Distance) {
		cout << "��Ա�����жϣ���p��Բc�ϣ�" << endl;
	}
	else if (rDistance > Distance) {
		cout << "��Ա�����жϣ���p��Բc�ڣ�" << endl;
	}
	else {
		cout << "��Ա�����жϣ���p��Բc�⣡" << endl;
	}
}