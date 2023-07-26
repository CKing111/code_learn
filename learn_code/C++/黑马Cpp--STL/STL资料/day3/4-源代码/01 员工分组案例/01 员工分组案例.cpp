#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <string>
#include <map>
#include <ctime>

//��˾������Ƹ��5��Ա����5��Ա�����빫˾֮����Ҫָ��Ա�����Ǹ����Ź���
//��Ա��Ϣ��: ���� ���� �绰 ���ʵ����
//ͨ��Multimap������Ϣ�Ĳ��� ���� ��ʾ
//�ֲ�����ʾԱ����Ϣ ��ʾȫ��Ա����Ϣ

enum
{
	CAIWU,RENLI,MEISHU
};

class Worker
{
public:
	string m_Name;//����
	int m_Money; //����
};

void createWorker(vector<Worker>&v)
{
	string nameSeed = "ABCDE";
	for (int i = 0; i < 5;i++)
	{
		Worker worker;
		worker.m_Name = "Ա��";
		worker.m_Name += nameSeed[i];

		worker.m_Money = rand() % 10000 + 10000; // 10000 ~ 19999
		
		v.push_back(worker);
	}

}

void setGroup(vector<Worker>&v, multimap<int, Worker>&m)
{
	for (vector<Worker>::iterator it = v.begin(); it != v.end();it++)
	{
		//����������ű��  0 1 2 
		int id = rand() % 3;

		//��Ա�����뵽�����������
		m.insert(make_pair(id, *it));
	}

}

void showWorker(multimap<int,Worker>&m)
{
	// 0 A   0  B   1  C   2  D  2 E
	cout << "��������Ա���£� " << endl;
	multimap<int,Worker>::iterator pos = m.find(CAIWU);
	int count = m.count(CAIWU);
	int index = 0;

	for (; pos != m.end(), index < count; pos++, index++)
	{
		cout << "������ " << pos->second.m_Name << " ���ʣ� " << pos->second.m_Money << endl;
	}

	cout << "������Դ������Ա���£� " << endl;
	pos = m.find(RENLI);
	count = m.count(RENLI);
	index = 0;

	for (; pos != m.end(), index < count; pos++, index++)
	{
		cout << "������ " << pos->second.m_Name << " ���ʣ� " << pos->second.m_Money << endl;
	}



	cout << "����������Ա���£� " << endl;
	pos = m.find(MEISHU);
	count = m.count(MEISHU);
	index = 0;

	for (; pos != m.end(), index < count; pos++, index++)
	{
		cout << "������ " << pos->second.m_Name << " ���ʣ� " << pos->second.m_Money << endl;
	}

}

int main(){

	//���������
	srand((unsigned int)time(NULL));

	vector<Worker>v; //���Ա��������
	//Ա���Ĵ���
	createWorker(v);

	//Ա������
	multimap<int, Worker>m;
	setGroup(v,m);

	//�ֲ�����ʾԱ��
	showWorker(m);


	//for (vector<Worker>::iterator it = v.begin(); it != v.end();it++)
	//{
	//	cout << "������ " << it->m_Name << " ���ʣ� " << it->m_Money << endl;
	//}


	system("pause");
	return EXIT_SUCCESS;
}