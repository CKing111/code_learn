#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include<vector>
#include<deque>
#include<algorithm>
#include<ctime>
using namespace std;


//��ί��ְ���(sort�㷨����)��10����ί��5��ѡ�ֽ��д��
//����5��ѡ��(�������÷�)���ŵ�vector�У�����vectorȡ��ÿ��ѡ�֣�ѭ����֣���10����ַ���deque�����У�
//������sort���д������
//�÷ֹ���ȥ����߷֣�ȥ����ͷ֣�ȡ��ƽ����
//���÷ֶ�5��ѡ�ֽ�������

class Person
{
public:
	Person(const string name,const int age)
	{
		this->m_Name = name;
		this->m_Age = age;
		cout << "����ѡ�֣�����:  " << m_Name << " ���䣺 " << m_Age << endl;
	}

	void finallScore(float score)
	{
		this->m_Score = score;
		cout << "ѡ�� " << this->m_Name << " �����յ÷�Ϊ��  " << this->m_Score << endl;
	}
	float m_Score;
	string m_Name;
	int m_Age;
};


class Judger
{
public:
	Judger(const string name)
	{
		this->m_Name = name;
		cout << "������ί������:  " << m_Name<< endl;
	}

	int gradeToPerson(const Person &p)
	{
		
		int score = rand()%11; //0-10��֮��
		cout << "��ί " << this->m_Name << " ���ڸ�ѡ�� " << p.m_Name << " ����,�������ķ����� "<< score <<endl;
		return score;
	}

	
	string m_Name;
};

float averageScore(const deque<int>& de)
{
	int totalScore=0;
	for (deque<int>::const_iterator d = de.begin(); d != de.end(); ++d)
	{
		totalScore += (*d);
	}
	//cout << "�ܷ֣�" << totalScore<<"������ "<<de.size() << endl;
	return totalScore / float(de.size());	
	
}

bool comparePerson(Person& p1, Person& p2)
{
	return p1.m_Score > p2.m_Score;
}
void test()
{
	vector<Judger> jude_vec;//������ί
	jude_vec.reserve(10);
	for (int i=1;i<=10;++i)
	{
		jude_vec.push_back(Judger(to_string(i)));
	}

	vector<Person> person_vec;//����ѡ��
	person_vec.reserve(5);
	person_vec.push_back(Person("a", 8));
	person_vec.push_back(Person("b", 9));
	person_vec.push_back(Person("b", 10));
	person_vec.push_back(Person("d", 8));
	person_vec.push_back(Person("e", 10));

	deque<int> d_Score;//����Ʒ���

	for (vector<Person>::iterator pv = person_vec.begin(); pv != person_vec.end(); ++pv)
	{
		for (vector<Judger>::iterator jv = jude_vec.begin(); jv != jude_vec.end(); ++jv)
		{
			d_Score.push_front((*jv).gradeToPerson(*pv));
			
			//cout << (*jv).m_Name << endl;		
		}
		sort(d_Score.begin(), d_Score.end());
		int mostHigh = d_Score.back();
		int lowHigh = d_Score.front();
		d_Score.pop_back();
		d_Score.pop_front();
		float average = averageScore(d_Score);
		cout << "ȥ��һ����߷֣�" << mostHigh << "ȥ��һ����ͷ֣�" << lowHigh << " ���յ÷�Ϊ��" << average << endl;
		(*pv).finallScore(average);
		d_Score.clear();	
	}

	sort(person_vec.begin(), person_vec.end(), comparePerson);
	cout << endl << "ѡ��������" << endl;
	for (vector<Person>::iterator pv = person_vec.begin(); pv != person_vec.end(); ++pv)
	{
		cout << "����:" << pv->m_Name << " �÷�:" << pv->m_Score << endl;
	}
	
}


int main()
{	
	srand((int)time(NULL));
	test();
	system("pause");
	return 0;
}