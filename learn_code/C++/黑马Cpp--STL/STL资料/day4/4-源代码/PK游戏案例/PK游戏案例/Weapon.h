#pragma  once
#include <iostream>
using namespace std;

//武器基类
class Weapon
{
public:
	//获取基础伤害
	virtual int getBaseDamage() = 0;
	//暴击效果 返回值大于0 触发暴击
	virtual int getCrit() = 0;
	//获取吸血 返回值大于0 触发吸血
	virtual int getSuckBlood() = 0;
	//冰冻效果 返回true 代表冰冻
	virtual int getFrozen() = 0;

	//触发概率
	virtual bool isTrigger(int rate) = 0;

public:
	int baseDamage; //基础攻击
	string weaponName; //武器名称
	int critPlus;  //暴击系数
	int critRate;  //暴击率
	int suckPlus;  //吸血系数
	int suckRate;  //吸血率
	int frozenRate;  //冰冻率
};
