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
1) 产生选手 （ ABCDEFGHIJKLMNOPQRSTUVWX ） 姓名、得分；选手编号
2) 第1轮	选手抽签 选手比赛 查看比赛结果
3) 第2轮	选手抽签 选手比赛 查看比赛结果
4) 第3轮	选手抽签 选手比赛 查看比赛结果
*/

class Speaker
{
public:
	string m_Name; //姓名
	int m_Score[3]; //选手得分
};


void createSpeaker(vector<int>&v, map<int,Speaker>&m)
{
	string nameSeed = "ABCDEFGHIJKLMNOPQRSTUVWX";

	for (int i = 0; i < 24;i++)
	{
		string name = "选手";
		name += nameSeed[i];

		Speaker sp;
		sp.m_Name = name;

		for (int j = 0; j < 3;j++)
		{
			sp.m_Score[j] = 0;
		}

		//v存选手编号  100 ~ 123
		v.push_back(i + 100);

		//m存放编号 和对应选手
		m.insert(make_pair(i + 100, sp));
	}
}


void speechDraw(vector<int>&v)
{
	random_shuffle(v.begin(), v.end());
}

// 参数1  代表 比赛轮数  参数2  比赛人员编号  参数3  人员编号和具体人员信息 参数4   晋级人员编号
void speechContest(int index, vector<int>&v ,  map<int,Speaker>&m , vector<int>&v2)
{
	multimap<int, int, greater<int>> mGroup; //临时容器  存放小组人员信息， key代表分数、value代表编号，greater代表排序规则
	int num = 0;
	for (vector<int>::iterator it = v.begin(); it != v.end();it++)
	{
		num++;
		//deque容器 存放评委打分
		deque<int>d;
		for (int i = 0; i < 10;i++)
		{
			int score = rand() % 41 + 60; //60 ~ 100
			d.push_back(score);
		}

		//排序
		sort(d.begin(), d.end());

		//去除最高分和最低分
		d.pop_back();
		d.pop_front();

		//计算总分
		int sum = accumulate(d.begin(), d.end(), 0);

		//计算平均分
		int avg = sum / d.size();

		//将平均分 同步到 人身上
		m[*it].m_Score[index - 1] = avg;

		//将信息 存放到临时容器中
		mGroup.insert(make_pair(avg, *it));

		//每6个选手  取出前三名 晋级
		if (num%6 == 0)
		{
			/*cout << "小组比赛成绩如下：" << endl;
			for (multimap<int, int, greater<int>>::iterator mit = mGroup.begin(); mit != mGroup.end(); mit++)
			{
				cout << "编号： " << mit->second << " 姓名： " << m[mit->second].m_Name << " 得分： " << m[mit->second].m_Score[index - 1] << endl;
			}*/

			//取前三名晋级
			int count = 0;
			for (multimap<int, int, greater<int>>::iterator mit = mGroup.begin(); mit != mGroup.end(), count < 3; mit++, count++)
			{
				//将前三名  存放到 v2容器中
				v2.push_back((*mit).second);
			}
			mGroup.clear();
		}

	}
}

void showScore(int index, vector<int>&v, map<int, Speaker>&m)
{
	cout << "第" << index << "轮比赛成绩如下： " << endl;

	for (map<int, Speaker>::iterator it = m.begin(); it != m.end();it++)
	{
		cout << "选手编号： " << it->first << " 姓名： " << it->second.m_Name << " 得分： " << it->second.m_Score[index - 1] << endl;
	}

	//晋级人员编号
	cout << "晋级人员编号如下： " << endl;
	for (vector<int>::iterator it = v.begin(); it != v.end();it++)
	{
		cout << *it << endl;
	}

}

int main(){

	//随机种子
	srand((unsigned int)time(NULL));

	vector<int>v; //存放选手编号容器
	map<int, Speaker> m; //存放选手以及选手对应的编号

	//1、创建选手
	createSpeaker(v,m);

	//2、抽签
	speechDraw(v);

	//3、选手比赛
	vector<int>v2; //第一轮晋级人员编号的容器
	speechContest(1, v, m, v2);

	//4、显示得分
	showScore(1,v2 , m);


	// 第二轮比赛
	speechDraw(v2);
	vector<int>v3; //第二轮晋级人员编号的容器
	speechContest(2, v2, m, v3);
	showScore(2, v3, m);


	//第三轮比赛
	speechDraw(v3);
	vector<int>v4; //第二轮晋级人员编号的容器
	speechContest(3, v3, m, v4);
	showScore(3, v4, m);



	//测试 
	//for (map<int, Speaker>::iterator it = m.begin(); it != m.end();it++)
	//{
	//	cout << "选手编号： " << it->first << " 姓名：" << it->second.m_Name << " 得分：" << it->second.m_Score[0] << endl;
	//}



	system("pause");
	return EXIT_SUCCESS;
}