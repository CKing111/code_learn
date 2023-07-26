#pragma  once
#include <iostream>
using namespace std;

//��������
class Weapon
{
public:
	//��ȡ�����˺�
	virtual int getBaseDamage() = 0;
	//����Ч�� ����ֵ����0 ��������
	virtual int getCrit() = 0;
	//��ȡ��Ѫ ����ֵ����0 ������Ѫ
	virtual int getSuckBlood() = 0;
	//����Ч�� ����true �������
	virtual int getFrozen() = 0;

	//��������
	virtual bool isTrigger(int rate) = 0;

public:
	int baseDamage; //��������
	string weaponName; //��������
	int critPlus;  //����ϵ��
	int critRate;  //������
	int suckPlus;  //��Ѫϵ��
	int suckRate;  //��Ѫ��
	int frozenRate;  //������
};
