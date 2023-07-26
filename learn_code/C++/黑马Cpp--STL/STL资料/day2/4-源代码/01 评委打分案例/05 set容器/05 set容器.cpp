#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <set> //set�� multiset  ��ͷ�ļ� 
#include <string>
/*
3.7.2.1 set���캯��
set<T> st;//setĬ�Ϲ��캯����
mulitset<T> mst; //multisetĬ�Ϲ��캯��:
set(const set &st);//�������캯��
3.7.2.2 set��ֵ����
set& operator=(const set &st);//���صȺŲ�����
swap(st);//����������������
3.7.2.3 set��С����
size();//����������Ԫ�ص���Ŀ
empty();//�ж������Ƿ�Ϊ��

3.7.2.4 set�����ɾ������
insert(elem);//�������в���Ԫ�ء�
clear();//�������Ԫ��
erase(pos);//ɾ��pos��������ָ��Ԫ�أ�������һ��Ԫ�صĵ�������
erase(beg, end);//ɾ������[beg,end)������Ԫ�� ��������һ��Ԫ�صĵ�������
erase(elem);//ɾ��������ֵΪelem��Ԫ�ء�
*/

void printSet(set<int>&s)
{
	for (set<int>::iterator it = s.begin(); it != s.end();it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}

void test01()
{
	set<int>s;
	s.insert(10);
	s.insert(30);
	s.insert(20);
	s.insert(50);
	s.insert(40);

	printSet(s);

	s.erase(10);

	printSet(s);
}


/*
3.7.2.5 set���Ҳ���
find(key);//���Ҽ�key�Ƿ����,�����ڣ����ظü���Ԫ�صĵ��������������ڣ�����set.end();
count(key);//���Ҽ�key��Ԫ�ظ���
lower_bound(keyElem);//���ص�һ��key>=keyElemԪ�صĵ�������
upper_bound(keyElem);//���ص�һ��key>keyElemԪ�صĵ�������
equal_range(keyElem);//����������key��keyElem��ȵ������޵�������������
*/

void test02()
{
	set<int>s;
	s.insert(10);
	s.insert(30);
	s.insert(20);
	s.insert(50);
	s.insert(40);

	set<int>::iterator pos = s.find(30);
	if (pos != s.end())
	{
		cout << "�ҵ���Ԫ�أ�" << *pos << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}


	//ͳ��  ����set���� ���Ҫô��0  Ҫô��1
	int num = s.count(10);
	cout << "10�ĸ���Ϊ�� " << num << endl;


	//lower_bound(keyElem);//���ص�һ��key>=keyElemԪ�صĵ�������

	set<int>::iterator res = s.lower_bound(30);
	
	if (res != s.end())
	{
		cout << "�ҵ�lower_bound��ֵΪ�� " << *res << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}

	//upper_bound(keyElem);//���ص�һ��key>keyElemԪ�صĵ�������
	res = s.upper_bound(30);
	if (res != s.end())
	{
		cout << "�ҵ�upper_bound��ֵΪ�� " << *res << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}


	//equal_range(keyElem);//����������key��keyElem��ȵ������޵�������������
	pair< set<int>::iterator, set<int>::iterator> it = s.equal_range(30);

	if (it.first != s.end())
	{
		cout << "�ҵ�equal_range�е� lower_bound��ֵΪ�� " << *(it.first) << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}

	if ( it.second != s.end() )
	{
		cout << "�ҵ�equal_range�е� upper_bound��ֵΪ�� " << *(it.second) << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}
}

//����������
void test03()
{
	//��һ������
	pair<string, int> p(string("Tom"), 18);

	cout << "������ " << p.first << " ���䣺 " << p.second << endl;

	//�ڶ�������
	pair<string, int> p2 = make_pair("Jerry", 20);
	cout << "������ " << p2.first << " ���䣺 " << p2.second << endl;
}

void test04()//setr��������������ظ��ļ�ֵ
{
	set<int>s;

	pair<set<int>::iterator , bool> ret = s.insert(10);

	if (ret.second)
	{
		cout << "��һ������ɹ�" << endl;
	}
	else
	{
		cout << "��һ������δ�ɹ�" << endl;
	}

	ret = s.insert(10);

	if (ret.second)
	{
		cout << "�ڶ�������ɹ�" << endl;
	}
	else
	{
		cout << "�ڶ�������δ�ɹ�" << endl;
	}

	//printSet(s);

	multiset <int>ms;

	ms.insert(10);
	ms.insert(10);

	for (multiset<int>::iterator it = ms.begin(); it != ms.end();it++)
	{
		cout << *it << endl;
	}

}

//���÷º��� ָ��set�������������
class MyCompare
{
public:
	bool operator()(int v1 ,int v2)
	{
		return v1 > v2;
	}
};

void test05()
{
	set<int, MyCompare> s;

	s.insert(10);
	s.insert(30);
	s.insert(20);
	s.insert(50);
	s.insert(40);

	//Ĭ�������Ǵ�С����
	//printSet(s);

	//����֮ǰ ָ���������
	for (set<int, MyCompare>::iterator it = s.begin(); it != s.end();it++)
	{
		cout << *it << endl;
	}
}


class Person
{
public:
	Person(string name,int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}

	string m_Name;
	int m_Age;
};

class MyComparePerson
{
public:
	bool operator()(const  Person & p1 , const Person & p2)
	{
		//���� ����
		return p1.m_Age < p2.m_Age;
	}

};

void test06()
{
	set<Person, MyComparePerson> s;

	Person p1("aaa", 10);
	Person p2("bbb", 30);
	Person p3("ccc", 20);
	Person p4("ddd", 50);
	Person p5("eee", 40);

	s.insert(p1);
	s.insert(p2);
	s.insert(p3);
	s.insert(p4);
	s.insert(p5);

	//�����Զ����������ͣ�����ָ���������
	for (set<Person,MyComparePerson>::iterator it = s.begin(); it != s.end(); it++)
	{
		cout << "������ " << (*it).m_Name << " ���䣺 " << it->m_Age << endl;
	}

}

int main(){
	test01();
	//test02();
	//test03();
	//test04();
	//test05();
	//test06();

	system("pause");
	return EXIT_SUCCESS;
}