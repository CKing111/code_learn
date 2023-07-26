#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <queue>
#include <string>
/*
3.5.3.1 queue���캯��
queue<T> queT;//queue����ģ����ʵ�֣�queue�����Ĭ�Ϲ�����ʽ��
queue(const queue &que);//�������캯��
3.5.3.2 queue��ȡ�������ɾ������
push(elem);//����β���Ԫ��
pop();//�Ӷ�ͷ�Ƴ���һ��Ԫ��
back();//�������һ��Ԫ��
front();//���ص�һ��Ԫ��

3.5.3.3 queue��ֵ����
queue& operator=(const queue &que);//���صȺŲ�����
3.5.3.4 queue��С����
empty();//�ж϶����Ƿ�Ϊ��
size();//���ض��еĴ�С
*/

class Person
{
public:
	Person(string name, int age) :m_Name(name), m_Age(age)
	{}

	string m_Name;
	int m_Age;
};

void test01()
{
	queue<Person>Q;

	Person p1("aaa",10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);

	//���
	Q.push(p1);
	Q.push(p2);
	Q.push(p3);
	Q.push(p4);

	while ( !Q.empty())
	{
		//��ȡ��ͷԪ��
		Person pFront =  Q.front();
		cout << "��ͷԪ�� ������ " << pFront.m_Name << " ���䣺 " << pFront.m_Age << endl;

		//��ȡ��βԪ��
		Person pBack = Q.back();
		cout << "��βԪ�� ������ " << pBack.m_Name << " ���䣺 " << pBack.m_Age << endl;

		//����
		Q.pop();
	}

	cout << "���еĴ�СΪ�� " << Q.size() << endl;

}

int main(){

	test01();

	system("pause");
	return EXIT_SUCCESS;
}