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

	cout << "��ӭ�������������� " << endl;

	cout << "��ѡ������Ӣ�ۣ� " << endl;

	char buf[1024];
	sprintf(buf, "1��%s <%s>", mHeroData["1"]["heroName"].c_str(), mHeroData["1"]["heroInfo"].c_str());
	cout << buf << endl;

	sprintf(buf, "2��%s <%s>", mHeroData["2"]["heroName"].c_str(), mHeroData["2"]["heroInfo"].c_str());
	cout << buf << endl;

	sprintf(buf, "3��%s <%s>", mHeroData["3"]["heroName"].c_str(), mHeroData["3"]["heroInfo"].c_str());
	cout << buf << endl;

	int select = 0;

	cin >> select;  // 1 \n

	getchar(); //ȡ�߻��з�

	//ʵ����Ӣ��
	Hero hero(select);

	cout << "��ѡ���Ӣ���ǣ� " << hero.heroName << endl;

	cout << "��ѡ������������ " << endl;


	map<string, map<string, string>> mWeaponData;
	fm.loadCSVData("./Weapons.csv", mWeaponData);

	cout << "1�����ֿ�ȭ " << endl;

	sprintf(buf, "2��%s", mWeaponData["1"]["weaponName"].c_str());
	cout << buf << endl;

	sprintf(buf, "3��%s", mWeaponData["2"]["weaponName"].c_str());
	cout << buf << endl;

	sprintf(buf, "4��%s", mWeaponData["3"]["weaponName"].c_str());
	cout << buf << endl;


	cin >> select;

	getchar();

	Weapon * weapon = NULL;

	switch (select)
	{
	case 1:
		cout << "����ţ�ƣ�������" << endl;
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

	//װ������
	hero.EquipWeapon(weapon);

	//�����һ�� ����
	int id = 5;  //rand() % 5 + 1; // 1 ~ 5
	Monster monster(id);

	int round = 1; //�غ���
	while (true)
	{
		getchar();
		system("cls");
		cout << "--------- ��ǰ�ǵ� " << round << " �غ� -------- " << endl;
	
		//Ӣ�۹�������
		if (hero.heroHp <= 0)
		{
			cout << "Ӣ�� ��" << hero.heroName << " �ѹң���Ϸ������" << endl;
			break;
		}
		hero.Attack(&monster);

		//���ﷴ��Ӣ��
		if (monster.monsterHp <= 0)
		{
			cout << "���� ��" << monster.monsterName << "�ѹң���ϲ���أ�" << endl;
			break;
		}
		monster.Attack(&hero);

		cout << "Ӣ��" << hero.heroName << "ʣ��Ѫ��Ϊ�� " << hero.heroHp << endl;
		cout << "����" << monster.monsterName << "ʣ��Ѫ��Ϊ�� " << monster.monsterHp << endl;

		//Ӣ�۹�������
		if (hero.heroHp <= 0)
		{
			cout << "Ӣ�� ��" << hero.heroName << " �ѹң���Ϸ������" << endl;
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


	//cout << "1��Ӣ�������� " << mHeroData["1"]["heroName"] << endl;
	//cout << "2��Ӣ��HP�� " << mHeroData["2"]["heroHp"] << endl;  //300
	//cout << "3��Ӣ��ATK�� " << mHeroData["3"]["heroAtk"] << endl;  //25

	//cout << "1�Ź������ƣ�" << mMonsterData["1"]["monsterName"] << endl;
	//cout << "3�Ź������ƣ�" << mMonsterData["3"]["monsterName"] << endl;
	//cout << "4�Ź������ƣ�" << mMonsterData["5"]["monsterName"] << endl;

	//cout << "1�����������ʣ� " << mWeaponData["1"]["weaponCritRate"] << endl;
	//cout << "2��������Ѫϵ���� " << mWeaponData["2"]["weaponSuckPlus"] << endl;
	//cout << "3�����������ʣ� " << mWeaponData["3"]["weaponFrozenRate"] << endl;


	//Weapon * weapon = new Knife;

	//cout << "knife ��������" << weapon->weaponName << endl;
	//cout << "knife �˺� " << weapon->baseDamage << endl;

	//delete weapon;

	//weapon = new BroadSword;
	//cout << "BroadSword ��������" << weapon->weaponName << endl;
	//cout << "BroadSword �˺� " << weapon->baseDamage << endl;

	//delete weapon;

	//weapon = new DragonSword;
	//cout << "DragonSword ��������" << weapon->weaponName << endl;
	//cout << "DragonSword �˺� " << weapon->baseDamage << endl;


	Fighting();


	system("pause");
	return EXIT_SUCCESS;
}