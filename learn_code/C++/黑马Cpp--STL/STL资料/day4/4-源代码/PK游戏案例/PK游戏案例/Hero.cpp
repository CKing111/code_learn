#include "Hero.h"

Hero::Hero(int heroId)
{
	FileManager fm;
	map<string, map<string, string>> mHeroData;
	fm.loadCSVData("./Hero.csv", mHeroData);

	//int 转 string

	string tempId = std::to_string(heroId);
	string id = mHeroData[tempId]["heroId"];

	this->heroName = mHeroData[id]["heroName"]; //英雄姓名
	this->heroAtk =  atoi( mHeroData[id]["heroAtk"].c_str()) ;  //英雄攻击力  
	this->heroDef = atoi(mHeroData[id]["heroDef"].c_str());     //英雄防御力
	this->heroHp = atoi(mHeroData[id]["heroHp"].c_str());       //英雄血量
	this->heroInfo = mHeroData[id]["heroInfo"];					//英雄简介

	this->pWeapon = NULL; //英雄武器
}

void Hero::Attack(Monster *monster)
{
	int crit = 0; //暴击
	int suck = 0; //吸血
	int frozen = 0; //冰冻
	int damage = 0; //英雄对怪物伤害

	if (this->pWeapon == NULL)
	{
		damage = this->heroAtk; //英雄没有装备武器，说明赤手空拳
	}
	else
	{
		//基础伤害  (自身攻击力 + 武器攻击力)
		damage = this->heroAtk + this->pWeapon->getBaseDamage();

		//是否暴击
		crit = this->pWeapon->getCrit();

		//是否吸血
		suck = this->pWeapon->getSuckBlood();

		//是否冰冻
		frozen = this->pWeapon->getFrozen();
	}

	if (crit) //触发暴击
	{
		damage += crit; //攻击力 再次 加上 暴击加成伤害
		cout << "英雄的武器触发暴击效果，怪物 < " << monster->monsterName << " > 受到暴击伤害！" << endl;
	}
	if (suck) //触发吸血
	{
		cout << "英雄的武器触发吸血效果，英雄 < " << this->heroName << " > 增加血量： " << suck << endl;
	}
	if (frozen)
	{
		cout << "英雄的武器触发冰冻效果，怪物 < " << monster->monsterName << " > 停止攻击一回合" << endl;
	}

	//给怪物冰冻属性赋值
	monster->isFrozen = frozen;

	//计算对怪物真实伤害
	int trueDamage = damage - monster->monsterDef > 0 ? damage - monster->monsterDef : 1;

	//吸血
	this->heroHp += suck;

	monster->monsterHp -= trueDamage;

	cout << "英雄： " << this->heroName << " 攻击了怪物： " << monster->monsterName << "造成的伤害为： " << trueDamage << endl;
}

void Hero::EquipWeapon(Weapon *weapon)
{
	if (weapon == NULL)
	{
		return;
	}

	this->pWeapon = weapon; //维护用户选择的武器
	cout << "英雄 < " << this->heroName << " > 装备了武器： " << this->pWeapon->weaponName << "!" <<  endl;

}
