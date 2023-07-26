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
			//���һ�����ʽ�ȡ
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
	//���ļ�  ifstream
	ifstream ifs(filePath);

	if (!ifs.is_open())
	{
		cout << "�ļ���ʧ��" << endl;
		return;
	}

	string firstLine;
	ifs >> firstLine;

	//cout << firstLine << endl;

	//����һ������ ������ vector������
	vector<string>v;


	//��������������  ��װ��һ������
	parseLineToVector(firstLine, v);

	//for (vector<string>::iterator it = v.begin(); it != v.end();it++)
	//{
	//	cout << *it << endl;
	//}

	//���map����׼��
	//map<string, map<string, string>> mapFileData;

	string data; //��������
	while (ifs >> data )
	{
		//cout << data << endl;
		vector<string> vOtherData; //�������� ��ŵ�vOtherData������
		parseLineToVector(data, vOtherData);

		//׼����һ��С����
		map<string, string > mSmall;
		//ƴ��С�����е�����
		for (int i = 0; i < v.size();i++)
		{
			mSmall.insert(make_pair(v[i], vOtherData[i]));
		}

		//��С���� ���뵽����������
		mapFileData.insert(make_pair(vOtherData[0], mSmall));
	}


	//cout << "1��Ӣ�������� " << mapFileData["1"]["heroName"] << endl;
	//cout << "2��Ӣ��HP�� " << mapFileData["2"]["heroHp"] << endl;  //300
	//cout << "3��Ӣ��ATK�� " << mapFileData["3"]["heroAtk"] << endl;  //25
}
