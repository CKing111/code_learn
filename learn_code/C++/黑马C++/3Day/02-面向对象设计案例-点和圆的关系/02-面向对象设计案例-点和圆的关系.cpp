#include<iostream>

using namespace std;

/*
	02-���������ư���-���Բ�Ĺ�ϵ
	���һ��Բ���ࣨAdvCircle������һ�����ࣨPoint����������Բ�Ĺ�ϵ��
	����Բ������Ϊx0, y0, �뾶Ϊr���������Ϊx1, y1��
	1������Բ�ϣ�(x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) == r*r
	2������Բ�ڣ�(x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) < r*r
	3������Բ�⣺(x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) > r*r

	˼����
		Բ�ࣺ��Ա��Բ�ġ�Բ��
		���ࣺ��Ա��x,y��
*/
// ����
class Point {
private:
	int m_X;
	int m_Y;

public:
	void setX(int x) {
		m_X = x;
	}
	void setY(int y) {
		m_Y = y;
	}
	int getX() { return m_X; }
	int getY() { return m_Y; }
};


// Բ��
class Circle {
private:
	int m_R;	// �뾶
	Point m_Center; // Բ��
public:
	void setR(int r) { m_R = r; }	// ���ð뾶
	int getR() { return m_R; }		// ��ȡ�뾶
	void setCenter(Point p) { m_Center = p; }	// ����Բ��
	Point getCenter() { return m_Center; }		// ��ȡԲ��
	// 1.����Բ���Ա�����жϵ�Բ��ϵ
	void isInCircleByClass(Point &p) {
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
};

// 2.����ȫ�ֺ����жϵ��Բ�Ĺ�ϵ
void isInCircle(Circle& c, Point& p) {
	// ��ȡԲ�ĵ���ľ���ƽ��
	int Distance = ((c.getCenter().getX() - p.getX()) * (c.getCenter().getX() - p.getX())) + ((c.getCenter().getY() - p.getY()) * (c.getCenter().getY() - p.getY()));
	// ��ȡ�뾶ƽ��
	int rDistance = c.getR() * c.getR();

	// �Ա�λ�ù�ϵ
	if (rDistance == Distance) {
		cout << "ȫ�ֺ����жϣ���p��Բc�ϣ�" << endl;
	}
	else if (rDistance > Distance) {
		cout << "ȫ�ֺ����жϣ���p��Բc�ڣ�" << endl;
	}
	else {
		cout << "ȫ�ֺ����жϣ���p��Բc�⣡" << endl;
	}
}

void test01() {
	Point p1;
	p1.setX(10);
	p1.setY(10);

	Circle c1;
	Point center;
	center.setX(0);
	center.setY(0);
	c1.setCenter(center);
	c1.setR(10);

	// 2.ȫ�ֺ����жϵ�Բλ�ù�ϵ
	isInCircle(c1, p1);
	// 1.��Ա�����ж�
	c1.isInCircleByClass(p1);
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}