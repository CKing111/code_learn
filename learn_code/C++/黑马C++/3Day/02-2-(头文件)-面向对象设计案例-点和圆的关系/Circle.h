#pragma once
#include<iostream>
#include"Point.h"
using namespace std;
// Բ��
class Circle {
private:
	int m_R;	// �뾶
	Point m_Center; // Բ��
public:
	void setR(int r);	// ���ð뾶
	int getR();	// ��ȡ�뾶
	void setCenter(Point p);	// ����Բ��
	Point getCenter(); 		// ��ȡԲ��
	// 1.����Բ���Ա�����жϵ�Բ��ϵ
	void isInCircleByClass(Point& p);
};