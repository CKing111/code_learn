#include "FileManager.h"



void FileManager::parseLineToVector(string firstLine, vector<string>&v)
{
	int pos = 0;
	int start = 0;
	//heroId,heroName,heroHp,heroAtk,heroDef,heroInfo
	while (true)
	{

		int pos = firstLine.find(",", start);
		if (pos == -1)
		{
			//最后一个单词截取
			string tmp = firstLine.substr(start, firstLine.size() - start);
			v.push_back(tmp);
			break;
		}
		string tmp = firstLine.substr(start, pos - start);
		v.push_back(tmp);
		start = pos + 1;
	}
}

void FileManager::loadCSVData(string filePath, map<string, map<string, string>>& mapFileData)
{
	//读文件  ifstream
	ifstream ifs(filePath);

	if (!ifs.is_open())
	{
		cout << "文件打开失败" << endl;
		return;
	}

	string firstLine;
	ifs >> firstLine;

	//cout << firstLine << endl;

	//将第一行数据 解析到 vector容器中
	vector<string>v;


	//将解析单行数据  封装成一个函数
	parseLineToVector(firstLine, v);

	//for (vector<string>::iterator it = v.begin(); it != v.end();it++)
	//{
	//	cout << *it << endl;
	//}

	//最大map容器准备
	//map<string, map<string, string>> mapFileData;

	string data; //其他数据
	while (ifs >> data )
	{
		//cout << data << endl;
		vector<string> vOtherData; //其他数据 存放到vOtherData容器中
		parseLineToVector(data, vOtherData);

		//准备出一个小容器
		map<string, string > mSmall;
		//拼接小容器中的数据
		for (int i = 0; i < v.size();i++)
		{
			mSmall.insert(make_pair(v[i], vOtherData[i]));
		}

		//将小容器 放入到最大的容器中
		mapFileData.insert(make_pair(vOtherData[0], mSmall));
	}


	//cout << "1号英雄姓名： " << mapFileData["1"]["heroName"] << endl;
	//cout << "2号英雄HP： " << mapFileData["2"]["heroHp"] << endl;  //300
	//cout << "3号英雄ATK： " << mapFileData["3"]["heroAtk"] << endl;  //25
}
