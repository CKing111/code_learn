#include<iostream>

using namespace std;

/*
	02-面向对象设计案例-点和圆的关系
	设计一个圆形类（AdvCircle），和一个点类（Point），计算点和圆的关系。
	假如圆心坐标为x0, y0, 半径为r，点的坐标为x1, y1：
	1）点在圆上：(x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) == r*r
	2）点在圆内：(x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) < r*r
	3）点在圆外：(x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) > r*r

	思考：
		圆类：成员（圆心、圆）
		点类：成员（x,y）
*/
// 点类
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


// 圆类
class Circle {
private:
	int m_R;	// 半径
	Point m_Center; // 圆心
public:
	void setR(int r) { m_R = r; }	// 设置半径
	int getR() { return m_R; }		// 读取半径
	void setCenter(Point p) { m_Center = p; }	// 设置圆心
	Point getCenter() { return m_Center; }		// 读取圆心
	// 1.利用圆类成员函数判断点圆关系
	void isInCircleByClass(Point &p) {
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
};

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