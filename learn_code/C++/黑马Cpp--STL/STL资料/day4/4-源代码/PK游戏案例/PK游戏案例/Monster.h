#pragma  once
#include <iostream>
using namespace std;
#include "FileManager.h"
#include "Hero.h"

class Hero;
class Monster
{
public:
	Monster(int monsterId);
	void Attack(Hero *hero);
public:
	string monsterName;
	int monsterHp;
	int monsterAtk;
	int monsterDef;
	bool isFrozen;
};