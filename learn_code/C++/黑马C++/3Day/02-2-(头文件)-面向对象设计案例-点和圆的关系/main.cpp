#include<iostream>
#include"Circle.h"
#include"Point.h"

using namespace std;


// 2.利用全局函数判断点和圆的关系
void isInCircle(Circle& c, Point& p) {
	// 获取圆心到店的距离平方
	int Distance = ((c.getCenter().getX() - p.getX()) * (c.getCenter().getX() - p.getX())) + ((c.getCenter().getY() - p.getY()) * (c.getCenter().getY() - p.getY()));
	// 获取半径平方
	int rDistance = c.getR() * c.getR();

	// 对比位置关系
	if (rDistance == Distance) {
		cout << "全局函数判断：点p在圆c上！" << endl;
	}
	else if (rDistance > Distance) {
		cout << "全局函数判断：点p在圆c内！" << endl;
	}
	else {
		cout << "全局函数判断：点p在圆c外！" << endl;
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

	// 2.全局函数判断点圆位置关系
	isInCircle(c1, p1);
	// 1.成员函数判断
	c1.isInCircleByClass(p1);
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}