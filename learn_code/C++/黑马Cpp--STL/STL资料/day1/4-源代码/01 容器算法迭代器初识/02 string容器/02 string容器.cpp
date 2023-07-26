#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <string>
#include <stdexcept>
#include <vector>
/*
3.1.2.1 string ���캯��
string();//����һ���յ��ַ��� ����: string str;
string(const string& str);//ʹ��һ��string�����ʼ����һ��string����
string(const char* s);//ʹ���ַ���s��ʼ��
string(int n, char c);//ʹ��n���ַ�c��ʼ��

3.1.2.2 string������ֵ����
string& operator=(const char* s);//char*�����ַ��� ��ֵ����ǰ���ַ���
string& operator=(const string &s);//���ַ���s������ǰ���ַ���
string& operator=(char c);//�ַ���ֵ����ǰ���ַ���
string& assign(const char *s);//���ַ���s������ǰ���ַ���
string& assign(const char *s, int n);//���ַ���s��ǰn���ַ�������ǰ���ַ���
string& assign(const string &s);//���ַ���s������ǰ�ַ���
string& assign(int n, char c);//��n���ַ�c������ǰ�ַ���
string& assign(const string &s, int start, int n);//��s��start��ʼn���ַ���ֵ���ַ���
*/

void test01()
{
	//����
	string s1;
	string s2(s1); //��������
	string s3("aaa"); //�вι���
	string s4(10, 'c'); //�������� �вι���

	cout << s3 << endl;
	cout << s4 << endl;


	//��ֵ
	string s5;
	s5 = s4;
	//string& assign(const char *s, int n);//���ַ���s��ǰn���ַ�������ǰ���ַ���
	s5.assign("abcdefg", 3);
	cout << "s5 = " << s5 << endl;


	//string& assign(const string &s, int start, int n);//��s��start��ʼn���ַ���ֵ���ַ���
	// ��0��ʼ����
	string s6 = "abcdefg";
	string s7;
	s7.assign(s6, 3, 3);
	cout << "s7 =" << s7 << endl;


}



/*
3.1.2.3 string��ȡ�ַ�����
char& operator[](int n);//ͨ��[]��ʽȡ�ַ�
char& at(int n);//ͨ��at������ȡ�ַ�
*/

void test02()
{
	string s = "hello world";

	//for (int i = 0; i < s.size();i++)
	//{
	//	//cout << s[i] << endl;
	//	cout << s.at(i) << endl;
	//}
	 
	//at �� [] ����   []����Խ�� ֱ�ӹҵ�  ��  at����Խ�� ���׳�һ�� �쳣 out_of_range

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
3.1.2.4 stringƴ�Ӳ���
string& operator+=(const string& str);//����+=������
string& operator+=(const char* str);//����+=������
string& operator+=(const char c);//����+=������
string& append(const char *s);//���ַ���s���ӵ���ǰ�ַ�����β
string& append(const char *s, int n);//���ַ���s��ǰn���ַ����ӵ���ǰ�ַ�����β
string& append(const string &s);//ͬoperator+=()
string& append(const string &s, int pos, int n);//���ַ���s�д�pos��ʼ��n���ַ����ӵ���ǰ�ַ�����β
string& append(int n, char c);//�ڵ�ǰ�ַ�����β���n���ַ�c

3.1.2.5 string���Һ��滻
int find(const string& str, int pos = 0) const; //����str��һ�γ���λ��,��pos��ʼ����
int find(const char* s, int pos = 0) const;  //����s��һ�γ���λ��,��pos��ʼ����
int find(const char* s, int pos, int n) const;  //��posλ�ò���s��ǰn���ַ���һ��λ��
int find(const char c, int pos = 0) const;  //�����ַ�c��һ�γ���λ��
int rfind(const string& str, int pos = npos) const;//����str���һ��λ��,��pos��ʼ����
int rfind(const char* s, int pos = npos) const;//����s���һ�γ���λ��,��pos��ʼ����
int rfind(const char* s, int pos, int n) const;//��pos����s��ǰn���ַ����һ��λ��
int rfind(const char c, int pos = 0) const; //�����ַ�c���һ�γ���λ��
string& replace(int pos, int n, const string& str); //�滻��pos��ʼn���ַ�Ϊ�ַ���str
string& replace(int pos, int n, const char* s); //�滻��pos��ʼ��n���ַ�Ϊ�ַ���s
*/

void test03()
{
	//�ַ���ƴ��
	string  str1 = "��";
	string str2 = "������";

	str1 += str2;

	cout << str1 << endl;


	string str3 = "�찲��";

	str1.append(str3);

	cout << str1 << endl;


	//�ַ�������
	string str4 = "abcdefghide";
	int pos = str4.find("de"); //����Ҳ����Ӵ� ���� -1 ���ҵ����ص�һ�γ��ֵ�λ��
	//rfind�����������
	//�ڶ������� ��Ĭ����ʼ���ҵ�λ��  Ĭ����0
	cout << "pos = " << pos << endl;

	//string& replace(int pos, int n, const string& str); //�滻��pos��ʼn���ַ�Ϊ�ַ���str
	str4.replace(1, 3, "111111"); // a111111efg..
	cout << "str4 " << str4 << endl;
	
}


/*
3.1.2.6 string�Ƚϲ���
compare������>ʱ���� 1��<ʱ���� -1��==ʱ���� 0��
�Ƚ����ִ�Сд���Ƚ�ʱ�ο��ֵ�˳����Խǰ���ԽС��
��д��A��Сд��aС��
int compare(const string &s) const;//���ַ���s�Ƚ�
int compare(const char *s) const;//���ַ���s�Ƚ�
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
3.1.2.7 string�Ӵ�
string substr(int pos = 0, int n = npos) const;//������pos��ʼ��n���ַ���ɵ��ַ���
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

	//����  �� ��ַ�е�ÿ������ ����ȡ�� vector������
	vector<string>v;

	// www   itcast   com  cn

	int start = 0;

	while (true)
	{
		//www.itcast.com.cn
		int pos = str.find(".",start);

		if (pos == -1)
		{
			//�����һ�����ʽ�ȡ
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
3.1.2.8 string�����ɾ������
string& insert(int pos, const char* s); //�����ַ���
string& insert(int pos, const string& str); //�����ַ���
string& insert(int pos, int n, char c);//��ָ��λ�ò���n���ַ�c
string& erase(int pos, int n = npos);//ɾ����Pos��ʼ��n���ַ�
*/
void test07()
{
	string str = "hello";
	str.insert(1, "111");

	cout << "str = " << str << endl; // h111ello


	//����erase  ɾ���� 111
	str.erase(1, 3);
	cout << "str = " << str << endl;
}

/*
string��c-style�ַ���ת��
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

	doWork(str2); //��������  const char*  ������ʽ����ת��Ϊ  string

	//doWork2(s); //������ ���� �� string ��ʽ����ת��Ϊ const char *
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

	//a = '1'; //ԭ��a��b��ָ���ʧЧ��
	//b = '2';

	cout << s << endl;
	cout << (int*)s.c_str() << endl;

}

/*
дһ�������������ڲ���string�ַ����е�����Сд��ĸ����Ϊ��д��ĸ��
*/
void test10()
{
	string str = "abCDeFg";

	for (int i = 0; i < str.size();i++)
	{
		//Сдת��д
		//str[i] = toupper(str[i]);

		//��дתСд
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