#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <string>
#include <stdexcept>
#include <vector>
/*
3.1.2.1 string 构造函数
string();//创建一个空的字符串 例如: string str;
string(const string& str);//使用一个string对象初始化另一个string对象
string(const char* s);//使用字符串s初始化
string(int n, char c);//使用n个字符c初始化

3.1.2.2 string基本赋值操作
string& operator=(const char* s);//char*类型字符串 赋值给当前的字符串
string& operator=(const string &s);//把字符串s赋给当前的字符串
string& operator=(char c);//字符赋值给当前的字符串
string& assign(const char *s);//把字符串s赋给当前的字符串
string& assign(const char *s, int n);//把字符串s的前n个字符赋给当前的字符串
string& assign(const string &s);//把字符串s赋给当前字符串
string& assign(int n, char c);//用n个字符c赋给当前字符串
string& assign(const string &s, int start, int n);//将s从start开始n个字符赋值给字符串
*/

void test01()
{
	//构造
	string s1;
	string s2(s1); //拷贝构造
	string s3("aaa"); //有参构造
	string s4(10, 'c'); //两个参数 有参构造

	cout << s3 << endl;
	cout << s4 << endl;


	//赋值
	string s5;
	s5 = s4;
	//string& assign(const char *s, int n);//把字符串s的前n个字符赋给当前的字符串
	s5.assign("abcdefg", 3);
	cout << "s5 = " << s5 << endl;


	//string& assign(const string &s, int start, int n);//将s从start开始n个字符赋值给字符串
	// 从0开始计算
	string s6 = "abcdefg";
	string s7;
	s7.assign(s6, 3, 3);
	cout << "s7 =" << s7 << endl;


}



/*
3.1.2.3 string存取字符操作
char& operator[](int n);//通过[]方式取字符
char& at(int n);//通过at方法获取字符
*/

void test02()
{
	string s = "hello world";

	//for (int i = 0; i < s.size();i++)
	//{
	//	//cout << s[i] << endl;
	//	cout << s.at(i) << endl;
	//}
	 
	//at 和 [] 区别   []访问越界 直接挂掉  而  at访问越界 会抛出一个 异常 out_of_range

	try
	{
		//s[100];
		s.at(100);
	}
	catch (exception &e)
	{
		cout << e.what() << endl;
	}

}


/*
3.1.2.4 string拼接操作
string& operator+=(const string& str);//重载+=操作符
string& operator+=(const char* str);//重载+=操作符
string& operator+=(const char c);//重载+=操作符
string& append(const char *s);//把字符串s连接到当前字符串结尾
string& append(const char *s, int n);//把字符串s的前n个字符连接到当前字符串结尾
string& append(const string &s);//同operator+=()
string& append(const string &s, int pos, int n);//把字符串s中从pos开始的n个字符连接到当前字符串结尾
string& append(int n, char c);//在当前字符串结尾添加n个字符c

3.1.2.5 string查找和替换
int find(const string& str, int pos = 0) const; //查找str第一次出现位置,从pos开始查找
int find(const char* s, int pos = 0) const;  //查找s第一次出现位置,从pos开始查找
int find(const char* s, int pos, int n) const;  //从pos位置查找s的前n个字符第一次位置
int find(const char c, int pos = 0) const;  //查找字符c第一次出现位置
int rfind(const string& str, int pos = npos) const;//查找str最后一次位置,从pos开始查找
int rfind(const char* s, int pos = npos) const;//查找s最后一次出现位置,从pos开始查找
int rfind(const char* s, int pos, int n) const;//从pos查找s的前n个字符最后一次位置
int rfind(const char c, int pos = 0) const; //查找字符c最后一次出现位置
string& replace(int pos, int n, const string& str); //替换从pos开始n个字符为字符串str
string& replace(int pos, int n, const char* s); //替换从pos开始的n个字符为字符串s
*/

void test03()
{
	//字符串拼接
	string  str1 = "我";
	string str2 = "爱北京";

	str1 += str2;

	cout << str1 << endl;


	string str3 = "天安门";

	str1.append(str3);

	cout << str1 << endl;


	//字符串查找
	string str4 = "abcdefghide";
	int pos = str4.find("de"); //如果找不到子串 返回 -1 ，找到返回第一次出现的位置
	//rfind从右往左查找
	//第二个参数 是默认起始查找的位置  默认是0
	cout << "pos = " << pos << endl;

	//string& replace(int pos, int n, const string& str); //替换从pos开始n个字符为字符串str
	str4.replace(1, 3, "111111"); // a111111efg..
	cout << "str4 " << str4 << endl;
	
}


/*
3.1.2.6 string比较操作
compare函数在>时返回 1，<时返回 -1，==时返回 0。
比较区分大小写，比较时参考字典顺序，排越前面的越小。
大写的A比小写的a小。
int compare(const string &s) const;//与字符串s比较
int compare(const char *s) const;//与字符串s比较
*/

void test04()
{

	string str1 = "bbcde";
	string str2 = "abcdeff";

	if (str1.compare(str2) == 0)
	{
		cout << "str1 == str2 " << endl;
	}
	else if (str1.compare(str2) > 0)
	{
		cout << "str1 > str2 " << endl;
	}
	else
	{
		cout << "str1 < str2 " << endl;
	}
}

/*
3.1.2.7 string子串
string substr(int pos = 0, int n = npos) const;//返回由pos开始的n个字符组成的字符串
*/
void test05()
{
	//string str = "abcde";
	//string subStr = str.substr(1, 3);
	//cout << subStr << endl; // bcd


	string email = "zhangtao@sina.com";
	int pos = email.find("@"); // 8
	string userName = email.substr(0, pos);

	cout << userName << endl;
}

void test06()
{
	string str = "www.itcast.com.cn";

	//需求：  将 网址中的每个单词 都截取到 vector容器中
	vector<string>v;

	// www   itcast   com  cn

	int start = 0;

	while (true)
	{
		//www.itcast.com.cn
		int pos = str.find(".",start);

		if (pos == -1)
		{
			//将最后一个单词截取
			string tmp = str.substr(start, str.size() - start);
			v.push_back(tmp);
			break;
		}

		string tmp = str.substr(start, pos- start);

		v.push_back(tmp);

		start = pos + 1;
	}

	
	for (vector<string>::iterator it = v.begin(); it != v.end();it++)
	{
		cout << *it << endl;
	}
}

/*
3.1.2.8 string插入和删除操作
string& insert(int pos, const char* s); //插入字符串
string& insert(int pos, const string& str); //插入字符串
string& insert(int pos, int n, char c);//在指定位置插入n个字符c
string& erase(int pos, int n = npos);//删除从Pos开始的n个字符
*/
void test07()
{
	string str = "hello";
	str.insert(1, "111");

	cout << "str = " << str << endl; // h111ello


	//利用erase  删除掉 111
	str.erase(1, 3);
	cout << "str = " << str << endl;
}

/*
string和c-style字符串转换
*/

void doWork(string s)
{
}

void doWork2(const char * s)
{
}

void test08()
{
	//char *  ->string
	char * str = "hello";
	string s(str);


	// string -> char *
	const char * str2 =  s.c_str();

	doWork(str2); //编译器将  const char*  可以隐式类型转换为  string

	//doWork2(s); //编译器 不会 将 string 隐式类型转换为 const char *
}


void test09()
{
	string s = "abcdefg";
	char& a = s[2];
	char& b = s[3];

	a = '1';
	b = '2';

	cout << s << endl;
	cout << (int*)s.c_str() << endl;

	s = "pppppppppppppppppppppppp";

	//a = '1'; //原来a和b的指向就失效了
	//b = '2';

	cout << s << endl;
	cout << (int*)s.c_str() << endl;

}

/*
写一个函数，函数内部将string字符串中的所有小写字母都变为大写字母。
*/
void test10()
{
	string str = "abCDeFg";

	for (int i = 0; i < str.size();i++)
	{
		//小写转大写
		//str[i] = toupper(str[i]);

		//大写转小写
		str[i] = tolower(str[i]);

	}
	cout << str << endl;
}

int main(){
	//test01();
	//test02();
	//test03();
	//test04();
	//test05();
	//test06();
	//test07();
	//test09();
	test10();

	system("pause");
	return EXIT_SUCCESS;
}