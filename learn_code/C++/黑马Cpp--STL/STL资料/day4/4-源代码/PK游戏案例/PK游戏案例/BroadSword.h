#pragma  once
#include <iostream>
using namespace std;
#include "Weapon.h"
#include "FileManager.h"

class BroadSword :public Weapon
{
public:
	BroadSword();

	//��ȡ�����˺�
	virtual int getBaseDamage();
	//����Ч�� ����ֵ����0 ��������
	virtual int getCrit();
	//��ȡ��Ѫ ����ֵ����0 ������Ѫ
	virtual int getSuckBlood();
	//����Ч�� ����true �������
	virtual int getFrozen();

	//��������
	virtual bool isTrigger(int rate);

};