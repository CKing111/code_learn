#include "Hero.h"

Hero::Hero(int heroId)
{
	FileManager fm;
	map<string, map<string, string>> mHeroData;
	fm.loadCSVData("./Hero.csv", mHeroData);

	//int ת string

	string tempId = std::to_string(heroId);
	string id = mHeroData[tempId]["heroId"];

	this->heroName = mHeroData[id]["heroName"]; //Ӣ������
	this->heroAtk =  atoi( mHeroData[id]["heroAtk"].c_str()) ;  //Ӣ�۹�����  
	this->heroDef = atoi(mHeroData[id]["heroDef"].c_str());     //Ӣ�۷�����
	this->heroHp = atoi(mHeroData[id]["heroHp"].c_str());       //Ӣ��Ѫ��
	this->heroInfo = mHeroData[id]["heroInfo"];					//Ӣ�ۼ��

	this->pWeapon = NULL; //Ӣ������
}

void Hero::Attack(Monster *monster)
{
	int crit = 0; //����
	int suck = 0; //��Ѫ
	int frozen = 0; //����
	int damage = 0; //Ӣ�۶Թ����˺�

	if (this->pWeapon == NULL)
	{
		damage = this->heroAtk; //Ӣ��û��װ��������˵�����ֿ�ȭ
	}
	else
	{
		//�����˺�  (�������� + ����������)
		damage = this->heroAtk + this->pWeapon->getBaseDamage();

		//�Ƿ񱩻�
		crit = this->pWeapon->getCrit();

		//�Ƿ���Ѫ
		suck = this->pWeapon->getSuckBlood();

		//�Ƿ����
		frozen = this->pWeapon->getFrozen();
	}

	if (crit) //��������
	{
		damage += crit; //������ �ٴ� ���� �����ӳ��˺�
		cout << "Ӣ�۵�������������Ч�������� < " << monster->monsterName << " > �ܵ������˺���" << endl;
	}
	if (suck) //������Ѫ
	{
		cout << "Ӣ�۵�����������ѪЧ����Ӣ�� < " << this->heroName << " > ����Ѫ���� " << suck << endl;
	}
	if (frozen)
	{
		cout << "Ӣ�۵�������������Ч�������� < " << monster->monsterName << " > ֹͣ����һ�غ�" << endl;
	}

	//������������Ը�ֵ
	monster->isFrozen = frozen;

	//����Թ�����ʵ�˺�
	int trueDamage = damage - monster->monsterDef > 0 ? damage - monster->monsterDef : 1;

	//��Ѫ
	this->heroHp += suck;

	monster->monsterHp -= trueDamage;

	cout << "Ӣ�ۣ� " << this->heroName << " �����˹�� " << monster->monsterName << "��ɵ��˺�Ϊ�� " << trueDamage << endl;
}

void Hero::EquipWeapon(Weapon *weapon)
{
	if (weapon == NULL)
	{
		return;
	}

	this->pWeapon = weapon; //ά���û�ѡ�������
	cout << "Ӣ�� < " << this->heroName << " > װ���������� " << this->pWeapon->weaponName << "!" <<  endl;

}
