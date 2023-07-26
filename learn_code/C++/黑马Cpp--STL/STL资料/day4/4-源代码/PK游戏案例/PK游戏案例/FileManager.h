#pragma once 
#include <iostream>
using namespace std;
#include <fstream>
#include <string>
#include <vector>
#include <map>

//文件管理类
class FileManager
{
public:

	//解析单行数据
	void parseLineToVector( string line, vector<string>&v );

	//加载CSV文件格式的函数
	void loadCSVData( string filePath,  map<string,map<string,string>>& m );
};