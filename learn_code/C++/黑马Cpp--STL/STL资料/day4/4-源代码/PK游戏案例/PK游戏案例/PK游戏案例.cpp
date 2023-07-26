#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include "FileManager.h"
#include "Weapon.h"
#include "Knife.h"
#include "BroadSword.h"
#include "DragonSword.h"
#include "Hero.h"
#include "Monster.h"
#include <ctime>

void Fighting()
{
	FileManager fm;
	map<string, map<string, string>> mHeroData;
	fm.loadCSVData("./Hero.csv", mHeroData);

	cout << "»¶Ó­À´µ½Á¦Á¿´óÈü£º " << endl;

	cout << "ÇëÑ¡ÔñÄúµÄÓ¢ÐÛ£º " << endl;

	char buf[1024];
	sprintf(buf, "1¡¢%s <%s>", mHeroData["1"]["heroName"].c_str(), mHeroData["1"]["heroInfo"].c_str());
	cout << buf << endl;

	sprintf(buf, "2¡¢%s <%s>", mHeroData["2"]["heroName"].c_str(), mHeroData["2"]["heroInfo"].c_str());
	cout << buf << endl;

	sprintf(buf, "3¡¢%s <%s>", mHeroData["3"]["heroName"].c_str(), mHeroData["3"]["heroInfo"].c_str());
	cout << buf << endl;

	int select = 0;

	cin >> select;  // 1 \n

	getchar(); //È¡×ß»»ÐÐ·û

	//ÊµÀý»¯Ó¢ÐÛ
	Hero hero(select);

	cout << "ÄúÑ¡ÔñµÄÓ¢ÐÛÊÇ£º " << hero.heroName << endl;

	cout << "ÇëÑ¡ÔñÄúµÄÎäÆ÷£º " << endl;


	map<string, map<string, string>> mWeaponData;
	fm.loadCSVData("./Weapons.csv", mWeaponData);

	cout << "1¡¢³àÊÖ¿ÕÈ­ " << endl;

	sprintf(buf, "2¡¢%s", mWeaponData["1"]["weaponName"].c_str());
	cout << buf << endl;

	sprintf(buf, "3¡¢%s", mWeaponData["2"]["weaponName"].c_str());
	cout << buf << endl;

	sprintf(buf, "4¡¢%s", mWeaponData["3"]["weaponName"].c_str());
	cout << buf << endl;


	cin >> select;

	getchar();

	Weapon * weapon = NULL;

	switch (select)
	{
	case 1:
		cout << "ÄãÕæÅ£±Æ£¬µÈËÀ°É" << endl;
		break;
	case  2:
		weapon = new Knife;
		break;
	case 3:
		weapon = new BroadSword;
		break;
	case 4:
		weapon = new DragonSword;
		break;
	default:
		break;
	}

	//×°±¸ÎäÆ÷
	hero.EquipWeapon(weapon);

	//Ëæ»ú³öÒ»¸ö ¹ÖÎï
	int id = 5;  //rand() % 5 + 1; // 1 ~ 5
	Monster monster(id);

	int round = 1; //»ØºÏÊý
	while (true)
	{
		getchar();
		system("cls");
		cout << "--------- µ±Ç°ÊÇµÚ " << round << " »ØºÏ -------- " << endl;
	
		//Ó¢ÐÛ¹¥»÷¹ÖÎï
		if (hero.heroHp <= 0)
		{
			cout << "Ó¢ÐÛ £º" << hero.heroName << " ÒÑ¹Ò£¬ÓÎÏ·½áÊø£¡" << endl;
			break;
		}
		hero.Attack(&monster);

		//¹ÖÎï·´»÷Ó¢ÐÛ
		if (monster.monsterHp <= 0)
		{
			cout << "¹ÖÎï £º" << monster.monsterName << "ÒÑ¹Ò£¬¹§Ï²¹ý¹Ø£¡" << endl;
			break;
		}
		monster.Attack(&hero);

		cout << "Ó¢ÐÛ" << hero.heroName << "Ê£ÓàÑªÁ¿Îª£º " << hero.heroHp << endl;
		cout << "¹ÖÎï" << monster.monsterName << "Ê£ÓàÑªÁ¿Îª£º " << monster.monsterHp << endl;

		//Ó¢ÐÛ¹¥»÷¹ÖÎï
		if (hero.heroHp <= 0)
		{
			cout << "Ó¢ÐÛ £º" << hero.heroName << " ÒÑ¹Ò£¬ÓÎÏ·½áÊø£¡" << endl;
			break;
		}

		round++;
	}
}


int main(){

	srand((unsigned int)time(NULL));

	//FileManager fm;
	//map<string, map<string, string>> mHeroData;
	//fm.loadCSVData("./Hero.csv", mHeroData);

	//map<string, map<string, string>> mMonsterData;
	//fm.loadCSVData("./Monsters.csv", mMonsterData);

	//map<string, map<string, string>> mWeaponData;
	//fm.loadCSVData("./Weapons.csv", mWeaponData);


	//cout << "1ºÅÓ¢ÐÛÐÕÃû£º " << mHeroData["1"]["heroName"] << endl;
	//cout << "2ºÅÓ¢ÐÛHP£º " << mHeroData["2"]["heroHp"] << endl;  //300
	//cout << "3ºÅÓ¢ÐÛATK£º " << mHeroData["3"]["heroAtk"] << endl;  //25

	//cout << "1ºÅ¹ÖÎïÃû³Æ£º" << mMonsterData["1"]["monsterName"] << endl;
	//cout << "3ºÅ¹ÖÎïÃû³Æ£º" << mMonsterData["3"]["monsterName"] << endl;
	//cout << "4ºÅ¹ÖÎïÃû³Æ£º" << mMonsterData["5"]["monsterName"] << endl;

	//cout << "1ºÅÎäÆ÷±©»÷ÂÊ£º " << mWeaponData["1"]["weaponCritRate"] << endl;
	//cout << "2ºÅÎäÆ÷ÎüÑªÏµÊý£º " << mWeaponData["2"]["weaponSuckPlus"] << endl;
	//cout << "3ºÅÎäÆ÷±ù¶³ÂÊ£º " << mWeaponData["3"]["weaponFrozenRate"] << endl;


	//Weapon * weapon = new Knife;

	//cout << "knife ÎäÆ÷Ãû³Æ" << weapon->weaponName << endl;
	//cout << "knife ÉËº¦ " << weapon->baseDamage << endl;

	//delete weapon;

	//weapon = new BroadSword;
	//cout << "BroadSword ÎäÆ÷Ãû³Æ" << weapon->weaponName << endl;
	//cout << "BroadSword ÉËº¦ " << weapon->baseDamage << endl;

	//delete weapon;

	//weapon = new DragonSword;
	//cout << "DragonSword ÎäÆ÷Ãû³Æ" << weapon->weaponName << endl;
	//cout << "DragonSword ÉËº¦ " << weapon->baseDamage << endl;


	Fighting();


	system("pause");
	return EXIT_SUCCESS;
}