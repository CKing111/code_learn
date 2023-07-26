#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector> //vector������ͷ�ļ�
#include <algorithm> //ϵͳ��׼�㷨ͷ�ļ�
#include <string>

//��ָͨ��Ҳ������һ�ֵ�����
void test01()
{
	int arr[5] = { 1, 5, 2, 7, 3 };
	int * p = arr;
	for (int i = 0; i < 5; i++)
	{
		//cout << arr[i] << endl;

		cout << *(p++) << endl;
	}
}

void myPrint(int val)
{
	cout << val << endl;
}

//������������
void test02()
{
	
	vector<int>v; //����һ��vector������ 

	//���������������
	v.push_back(10);
	v.push_back(20);
	v.push_back(30);
	v.push_back(40);

	//ͨ������������ ��������
	//ÿ�����������Լ�ר���ĵ�����
	//vector<int>::iterator itBegin = v.begin(); //��ʼ������

	//vector<int>::iterator itEnd = v.end(); //����������

	//��һ�ֱ�����ʽ
	//while (itBegin != itEnd)
	//{
	//	cout << *itBegin << endl;
	//	itBegin++;
	//}

	//�ڶ��ֱ�����ʽ
	//for (vector<int>::iterator it = v.begin(); it != v.end();it++)
	//{
	//	cout << *it << endl;
	//}

	//�����ֱ�����ʽ  ����ϵͳ�ṩ�㷨
	for_each(v.begin(), v.end(), myPrint);

}

//�Զ�����������
class Person
{
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}
	string m_Name;
	int m_Age;
};
void test03()
{
	vector<Person> v;

	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);

	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);

	//����
	for (vector<Person>::iterator it = v.begin(); it != v.end();it++)
	{
		// *it --- Person����    it  --- ָ��
		cout << "������ " << (*it).m_Name << " ���䣺 " << it->m_Age << endl;
	}
}

//����Զ����������͵�ָ��
void test04()
{
	vector<Person*> v;

	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);

	v.push_back(&p1);
	v.push_back(&p2);
	v.push_back(&p3);
	v.push_back(&p4);

	for (vector<Person*>::iterator it = v.begin(); it != v.end(); it++)
	{
		//*it   ---  Person *
		cout << "������ " << (*it)->m_Name << " ���䣺 " << (*it)->m_Age << endl;
	}
}

//����Ƕ������
void test05()
{
	vector< vector<int> > v ;//���ƶ�ά����

	vector<int>v1;
	vector<int>v2;
	vector<int>v3;

	for (int i = 0; i < 10;i++)
	{
		v1.push_back(i);
		v2.push_back(i + 10);
		v3.push_back(i + 100);
	}

	//��С���� ���뵽��������
	v.push_back(v1);
	v.push_back(v2);
	v.push_back(v3);


	for (vector<vector<int>>::iterator it = v.begin(); it != v.end();it++)
	{
		//*it --- vector<int>
		for (vector<int>::iterator vit = (*it).begin(); vit != (*it).end();vit++)
		{
			//*vit  --- int
			cout << *vit << " ";
		}
		cout << endl;
	}
}

int main(){
	//test01();
	//test02();
	//test03();
	//test04();
	test05();
	system("pause");
	return EXIT_SUCCESS;
}