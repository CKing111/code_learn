#pragma once 
#include <iostream>
using namespace std;
#include <fstream>
#include <string>
#include <vector>
#include <map>

//�ļ�������
class FileManager
{
public:

	//������������
	void parseLineToVector( string line, vector<string>&v );

	//����CSV�ļ���ʽ�ĺ���
	void loadCSVData( string filePath,  map<string,map<string,string>>& m );
};