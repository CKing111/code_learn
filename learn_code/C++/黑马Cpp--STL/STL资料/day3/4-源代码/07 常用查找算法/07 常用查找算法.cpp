#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
/*
find�㷨 ����Ԫ��
@param beg ������ʼ������
@param end ��������������
@param value ���ҵ�Ԫ��
@return ���ز���Ԫ�ص�λ��
*/

void test01()
{
	vector<int>v1;
	for (int i = 0; i < 10;i++)
	{
		v1.push_back(i);
	}

	//������û��5���Ԫ��
	vector<int>::iterator it = find(v1.begin(), v1.end(), 5);
	if (it != v1.end())
	{
		cout << "�ҵ���Ԫ�أ� " << *it << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}

}

class Person
{
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}

	bool operator==(const Person & p)
	{
		return this->m_Name == p.m_Name && this->m_Age == p.m_Age;
	}

	string m_Name;
	int m_Age;
};
void test02()
{
	vector<Person>v;
	Person p1("aaa", 10);
	Person p2("bbb", 40);
	Person p3("ccc", 20);
	Person p4("ddd", 30);

	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);

	vector<Person>::iterator it = find(v.begin(), v.end(), p3);
	if (it != v.end())
	{
		cout << "�ҵ���Ԫ��---������  " << (*it).m_Name  << " ���䣺 "<< it->m_Age << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}
}

class MyComparePerson :public binary_function<Person*,Person*,bool>
{
public:
	bool operator()(Person * p1 ,Person * p2) const
	{
		return (p1->m_Name == p2->m_Name && p1->m_Age == p2->m_Age);
	}
};

void test03()
{
	vector<Person*>v;
	Person p1("aaa", 10);
	Person p2("bbb", 40);
	Person p3("ccc", 20);
	Person p4("ddd", 30);

	v.push_back(&p1);
	v.push_back(&p2);
	v.push_back(&p3);
	v.push_back(&p4);

	Person * p = new Person("bbb", 40);

	vector<Person*>::iterator it= find_if(v.begin(), v.end(), bind2nd( MyComparePerson() ,p ));
	if (it != v.end())
	{
		cout << "�ҵ������� ������ " << (*it)->m_Name << " ���䣺 " << (*it)->m_Age << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}


}


/*
adjacent_find�㷨 ���������ظ�Ԫ��
@param beg ������ʼ������
@param end ��������������
@param  _callback �ص���������ν��(����bool���͵ĺ�������)
@return ��������Ԫ�صĵ�һ��λ�õĵ�����
*/
void test04()
{
	vector<int>v1;
	v1.push_back(1);
	v1.push_back(4);
	v1.push_back(2);
	v1.push_back(3);
	v1.push_back(4);
	v1.push_back(6);
	v1.push_back(6);
	v1.push_back(17);

	vector<int>::iterator it = adjacent_find(v1.begin(), v1.end());

	if (it != v1.end())
	{
		cout << "���ڵ��ظ�Ԫ��Ϊ��" << *it << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}


}



/*
binary_search�㷨 ���ֲ��ҷ�
ע��: �����������в�����
@param beg ������ʼ������
@param end ��������������
@param value ���ҵ�Ԫ��
@return bool ���ҷ���true ����false
*/

void test05()
{
	vector<int>v;
	for (int i = 0; i < 10;i++)
	{
		v.push_back(i);
	}
	//v.push_back(3);

	bool ret = binary_search(v.begin(), v.end(), 9);

	if (ret)
	{
		cout << "�ҵ���" << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}
}


/*
count�㷨 ͳ��Ԫ�س��ִ���
@param beg ������ʼ������
@param end ��������������
@param  value�ص���������ν��(����bool���͵ĺ�������)
@return int����Ԫ�ظ���

count_if�㷨 ͳ��Ԫ�س��ִ���
@param beg ������ʼ������
@param end ��������������
@param  callback �ص���������ν��(����bool���͵ĺ�������)
@return int����Ԫ�ظ���
*/
class MyCompare6
{
public:
	bool operator()(int val)
	{
		return val > 4;
	}
};
void test06()
{
	vector<int>v;
	for (int i = 0; i < 10; i++)
	{
		v.push_back(i);
	}
	v.push_back(4);
	v.push_back(4);
	v.push_back(4);
	v.push_back(4);

	int num = count(v.begin(), v.end(), 4);
	cout << "4�ĸ���Ϊ�� " << num << endl;


	//����������ͳ��
	num = count_if(v.begin(), v.end(), MyCompare6());
	cout << "����4�ĸ���Ϊ�� " << num << endl;
}



int main(){
	//test01();
	//test02();
	//test03();
	//test04();
	//test05();
	test06();
	system("pause");
	return EXIT_SUCCESS;
}