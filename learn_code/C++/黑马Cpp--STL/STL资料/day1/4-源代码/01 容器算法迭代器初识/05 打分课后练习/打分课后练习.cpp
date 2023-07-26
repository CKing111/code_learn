#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include<vector>
#include<deque>
#include<algorithm>
#include<ctime>
using namespace std;


//评委打分案例(sort算法排序)，10个评委对5个选手进行打分
//创建5个选手(姓名，得分)，放到vector中，遍历vector取出每个选手，循环打分，将10个打分放入deque容器中，
//，利用sort进行打分排序，
//得分规则：去除最高分，去除最低分，取出平均分
//按得分对5名选手进行排名

class Person
{
public:
	Person(const string name,const int age)
	{
		this->m_Name = name;
		this->m_Age = age;
		cout << "构造选手，姓名:  " << m_Name << " 年龄： " << m_Age << endl;
	}

	void finallScore(float score)
	{
		this->m_Score = score;
		cout << "选手 " << this->m_Name << " 的最终得分为：  " << this->m_Score << endl;
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
		cout << "构造评委，姓名:  " << m_Name<< endl;
	}

	int gradeToPerson(const Person &p)
	{
		
		int score = rand()%11; //0-10分之间
		cout << "评委 " << this->m_Name << " 正在给选手 " << p.m_Name << " 评分,他给出的分数是 "<< score <<endl;
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
	//cout << "总分：" << totalScore<<"人数： "<<de.size() << endl;
	return totalScore / float(de.size());	
	
}

bool comparePerson(Person& p1, Person& p2)
{
	return p1.m_Score > p2.m_Score;
}
void test()
{
	vector<Judger> jude_vec;//构造评委
	jude_vec.reserve(10);
	for (int i=1;i<=10;++i)
	{
		jude_vec.push_back(Judger(to_string(i)));
	}

	vector<Person> person_vec;//构造选手
	person_vec.reserve(5);
	person_vec.push_back(Person("a", 8));
	person_vec.push_back(Person("b", 9));
	person_vec.push_back(Person("b", 10));
	person_vec.push_back(Person("d", 8));
	person_vec.push_back(Person("e", 10));

	deque<int> d_Score;//构造计分器

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
		cout << "去掉一个最高分：" << mostHigh << "去掉一个最低分：" << lowHigh << " 最终得分为：" << average << endl;
		(*pv).finallScore(average);
		d_Score.clear();	
	}

	sort(person_vec.begin(), person_vec.end(), comparePerson);
	cout << endl << "选手排名：" << endl;
	for (vector<Person>::iterator pv = person_vec.begin(); pv != person_vec.end(); ++pv)
	{
		cout << "姓名:" << pv->m_Name << " 得分:" << pv->m_Score << endl;
	}
	
}


int main()
{	
	srand((int)time(NULL));
	test();
	system("pause");
	return 0;
}