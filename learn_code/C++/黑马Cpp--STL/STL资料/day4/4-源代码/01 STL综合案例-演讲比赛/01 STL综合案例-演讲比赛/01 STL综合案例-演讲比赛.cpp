#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <deque>
#include <numeric>
#include <functional>
#include <ctime>

/*
1) ����ѡ�� �� ABCDEFGHIJKLMNOPQRSTUVWX �� �������÷֣�ѡ�ֱ��
2) ��1��	ѡ�ֳ�ǩ ѡ�ֱ��� �鿴�������
3) ��2��	ѡ�ֳ�ǩ ѡ�ֱ��� �鿴�������
4) ��3��	ѡ�ֳ�ǩ ѡ�ֱ��� �鿴�������
*/

class Speaker
{
public:
	string m_Name; //����
	int m_Score[3]; //ѡ�ֵ÷�
};


void createSpeaker(vector<int>&v, map<int,Speaker>&m)
{
	string nameSeed = "ABCDEFGHIJKLMNOPQRSTUVWX";

	for (int i = 0; i < 24;i++)
	{
		string name = "ѡ��";
		name += nameSeed[i];

		Speaker sp;
		sp.m_Name = name;

		for (int j = 0; j < 3;j++)
		{
			sp.m_Score[j] = 0;
		}

		//v��ѡ�ֱ��  100 ~ 123
		v.push_back(i + 100);

		//m��ű�� �Ͷ�Ӧѡ��
		m.insert(make_pair(i + 100, sp));
	}
}


void speechDraw(vector<int>&v)
{
	random_shuffle(v.begin(), v.end());
}

// ����1  ���� ��������  ����2  ������Ա���  ����3  ��Ա��ź;�����Ա��Ϣ ����4   ������Ա���
void speechContest(int index, vector<int>&v ,  map<int,Speaker>&m , vector<int>&v2)
{
	multimap<int, int, greater<int>> mGroup; //��ʱ����  ���С����Ա��Ϣ�� key���������value�����ţ�greater�����������
	int num = 0;
	for (vector<int>::iterator it = v.begin(); it != v.end();it++)
	{
		num++;
		//deque���� �����ί���
		deque<int>d;
		for (int i = 0; i < 10;i++)
		{
			int score = rand() % 41 + 60; //60 ~ 100
			d.push_back(score);
		}

		//����
		sort(d.begin(), d.end());

		//ȥ����߷ֺ���ͷ�
		d.pop_back();
		d.pop_front();

		//�����ܷ�
		int sum = accumulate(d.begin(), d.end(), 0);

		//����ƽ����
		int avg = sum / d.size();

		//��ƽ���� ͬ���� ������
		m[*it].m_Score[index - 1] = avg;

		//����Ϣ ��ŵ���ʱ������
		mGroup.insert(make_pair(avg, *it));

		//ÿ6��ѡ��  ȡ��ǰ���� ����
		if (num%6 == 0)
		{
			/*cout << "С������ɼ����£�" << endl;
			for (multimap<int, int, greater<int>>::iterator mit = mGroup.begin(); mit != mGroup.end(); mit++)
			{
				cout << "��ţ� " << mit->second << " ������ " << m[mit->second].m_Name << " �÷֣� " << m[mit->second].m_Score[index - 1] << endl;
			}*/

			//ȡǰ��������
			int count = 0;
			for (multimap<int, int, greater<int>>::iterator mit = mGroup.begin(); mit != mGroup.end(), count < 3; mit++, count++)
			{
				//��ǰ����  ��ŵ� v2������
				v2.push_back((*mit).second);
			}
			mGroup.clear();
		}

	}
}

void showScore(int index, vector<int>&v, map<int, Speaker>&m)
{
	cout << "��" << index << "�ֱ����ɼ����£� " << endl;

	for (map<int, Speaker>::iterator it = m.begin(); it != m.end();it++)
	{
		cout << "ѡ�ֱ�ţ� " << it->first << " ������ " << it->second.m_Name << " �÷֣� " << it->second.m_Score[index - 1] << endl;
	}

	//������Ա���
	cout << "������Ա������£� " << endl;
	for (vector<int>::iterator it = v.begin(); it != v.end();it++)
	{
		cout << *it << endl;
	}

}

int main(){

	//�������
	srand((unsigned int)time(NULL));

	vector<int>v; //���ѡ�ֱ������
	map<int, Speaker> m; //���ѡ���Լ�ѡ�ֶ�Ӧ�ı��

	//1������ѡ��
	createSpeaker(v,m);

	//2����ǩ
	speechDraw(v);

	//3��ѡ�ֱ���
	vector<int>v2; //��һ�ֽ�����Ա��ŵ�����
	speechContest(1, v, m, v2);

	//4����ʾ�÷�
	showScore(1,v2 , m);


	// �ڶ��ֱ���
	speechDraw(v2);
	vector<int>v3; //�ڶ��ֽ�����Ա��ŵ�����
	speechContest(2, v2, m, v3);
	showScore(2, v3, m);


	//�����ֱ���
	speechDraw(v3);
	vector<int>v4; //�ڶ��ֽ�����Ա��ŵ�����
	speechContest(3, v3, m, v4);
	showScore(3, v4, m);



	//���� 
	//for (map<int, Speaker>::iterator it = m.begin(); it != m.end();it++)
	//{
	//	cout << "ѡ�ֱ�ţ� " << it->first << " ������" << it->second.m_Name << " �÷֣�" << it->second.m_Score[0] << endl;
	//}



	system("pause");
	return EXIT_SUCCESS;
}