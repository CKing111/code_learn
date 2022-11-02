#include<iostream>
#include"Circle.h"
#include"Point.h"

using namespace std;


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