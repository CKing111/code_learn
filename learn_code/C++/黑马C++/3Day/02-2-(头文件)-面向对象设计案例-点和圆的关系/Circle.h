#pragma once
#include<iostream>
#include"Point.h"
using namespace std;
// 圆类
class Circle {
private:
	int m_R;	// 半径
	Point m_Center; // 圆心
public:
	void setR(int r);	// 设置半径
	int getR();	// 读取半径
	void setCenter(Point p);	// 设置圆心
	Point getCenter(); 		// 读取圆心
	// 1.利用圆类成员函数判断点圆关系
	void isInCircleByClass(Point& p);
};