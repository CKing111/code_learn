#include "Monster.h"



Monster::Monster(int monsterId)
{
	FileManager fm;
	map<string, map<string, string>> mMonsterData;
	fm.loadCSVData("./Monsters.csv", mMonsterData);

	string tmpId = to_string(monsterId);
	string id = mMonsterData[tmpId]["monsterId"];


	this->monsterName = mMonsterData[id]["monsterName"]; // 怪物姓名
	this->monsterAtk = atoi( mMonsterData[id]["monsterAtk"].c_str()); //怪物攻击力
	this->monsterDef = atoi(mMonsterData[id]["monsterDef"].c_str()); //怪物防御力
	this->monsterHp = atoi(mMonsterData[id]["monsterHp"].c_str()); //怪物血量
	this->isFrozen = false; // 是否被冰冻 ，默认没有被冻上

}

void Monster::Attack(Hero *hero)
{
	//判断 怪物 是否被冰冻 ，如果被冰冻，停止攻击一回合
	if (this->isFrozen)
	{
		cout << "怪物： " << this->monsterName << "被冰冻了，本回合无法攻击英雄！" << endl;
		return;
	}

	//计算对英雄伤害
	int damage = this->monsterAtk - hero->heroDef > 0 ? this->monsterAtk - hero->heroDef : 1;

	//英雄掉血
	hero->heroHp -= damage;

	cout << "怪物 < " << this->monsterName << " > 攻击了英雄 < " << hero->heroName << " > ,造成的伤害为：" << damage << endl;

}
