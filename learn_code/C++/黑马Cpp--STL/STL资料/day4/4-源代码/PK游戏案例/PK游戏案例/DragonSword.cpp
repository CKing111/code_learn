#include "DragonSword.h"

//��ʼ������  ����
DragonSword::DragonSword()
{
	FileManager fm;
	map<string, map<string, string>> mWeaponData;
	fm.loadCSVData("./Weapons.csv", mWeaponData);

	string id = mWeaponData["3"]["weaponId"];
	this->weaponName = mWeaponData[id]["weaponName"]; //��������
	// string ת int
	this->baseDamage = atoi(mWeaponData[id]["weaponAtk"].c_str());  //����������
	this->critPlus = atoi(mWeaponData[id]["weaponCritPlus"].c_str()); //����ϵ��
	this->critRate = atoi(mWeaponData[id]["weaponCritRate"].c_str()); //������
	this->suckPlus = atoi(mWeaponData[id]["weaponSuckPlus"].c_str()); //��Ѫϵ��
	this->suckRate = atoi(mWeaponData[id]["weaponSuckRate"].c_str()); //��Ѫ��
	this->frozenRate = atoi(mWeaponData[id]["weaponFrozenRate"].c_str()); //������
}

int DragonSword::getBaseDamage()
{
	return this->baseDamage;
}

int DragonSword::getCrit()
{
	if (isTrigger(this->critRate))
	{
		//�����������  ���ػ����˺� *  ����ϵ��
		return this->baseDamage * this->critPlus;
	}
	else
	{
		return  0;
	}
}

int DragonSword::getSuckBlood()
{
	if (isTrigger(this->suckRate))
	{
		return this->baseDamage * this->suckPlus;
	}
	else
	{
		return 0;
	}
}

int DragonSword::getFrozen()
{
	if (isTrigger(this->frozenRate))
	{
		return 1;
	}
	else
	{
		return  0;
	}
}

bool DragonSword::isTrigger(int rate)
{
	int num = rand() % 100 + 1; // 1 ~ 100
	if (num < rate)
	{
		return true;
	}
	return false;
}
