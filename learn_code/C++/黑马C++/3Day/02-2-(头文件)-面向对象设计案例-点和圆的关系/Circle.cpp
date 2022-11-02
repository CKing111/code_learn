#include"Circle.h"

void Circle::setR(int r) { m_R = r; }	// 设置半径
int Circle::getR() { return m_R; }		// 读取半径
void Circle::setCenter(Point p) { m_Center = p; }	// 设置圆心
Point Circle::getCenter() { return m_Center; }		// 读取圆心
// 1.利用圆类成员函数判断点圆关系
void Circle::isInCircleByClass(Point& p) {
	// 获取圆心到店的距离平方
	int Distance = ((m_Center.getX() - p.getX()) * (m_Center.getX() - p.getX())) + ((m_Center.getY() - p.getY()) * (m_Center.getY() - p.getY()));
	// 获取半径平方
	int rDistance = m_R * m_R;
	// 对比位置关系
	if (rDistance == Distance) {
		cout << "成员函数判断：点p在圆c上！" << endl;
	}
	else if (rDistance > Distance) {
		cout << "成员函数判断：点p在圆c内！" << endl;
	}
	else {
		cout << "成员函数判断：点p在圆c外！" << endl;
	}
}