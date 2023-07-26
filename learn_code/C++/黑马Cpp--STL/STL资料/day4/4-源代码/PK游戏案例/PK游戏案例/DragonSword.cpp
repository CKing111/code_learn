#include "DragonSword.h"

//初始化砍刀  属性
DragonSword::DragonSword()
{
	FileManager fm;
	map<string, map<string, string>> mWeaponData;
	fm.loadCSVData("./Weapons.csv", mWeaponData);

	string id = mWeaponData["3"]["weaponId"];
	this->weaponName = mWeaponData[id]["weaponName"]; //武器名称
	// string 转 int
	this->baseDamage = atoi(mWeaponData[id]["weaponAtk"].c_str());  //基础攻击力
	this->critPlus = atoi(mWeaponData[id]["weaponCritPlus"].c_str()); //暴击系数
	this->critRate = atoi(mWeaponData[id]["weaponCritRate"].c_str()); //暴击率
	this->suckPlus = atoi(mWeaponData[id]["weaponSuckPlus"].c_str()); //吸血系数
	this->suckRate = atoi(mWeaponData[id]["weaponSuckRate"].c_str()); //吸血率
	this->frozenRate = atoi(mWeaponData[id]["weaponFrozenRate"].c_str()); //冰冻率
}

int DragonSword::getBaseDamage()
{
	return this->baseDamage;
}

int DragonSword::getCrit()
{
	if (isTrigger(this->critRate))
	{
		//如果触发暴击  返回基础伤害 *  暴击系数
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
