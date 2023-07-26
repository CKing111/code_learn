#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <algorithm>
#include<list>
/*
list��һ��˫��ѭ������
*/
/**/
//void test01()
//{
//	list<int> myList;
//	for (int i = 0; i < 10; i++){
//		myList.push_back(i);
//	}
//	list<int>::_Nodeptr node = myList._Myhead->_Next;
//	for (int i = 0; i < myList._Mysize * 2; i++){
//		cout << "Node:" << node->_Myval << endl;
//		node = node->_Next;
//		//node->_Prev 
//		if (node == myList._Myhead){
//			node = node->_Next;
//		}
//	}
//}

/*
3.6.4.1 list���캯��
list<T> lstT;//list���ò���ģ����ʵ��,�����Ĭ�Ϲ�����ʽ��
list(beg,end);//���캯����[beg, end)�����е�Ԫ�ؿ���������
list(n,elem);//���캯����n��elem����������
list(const list &lst);//�������캯����
3.6.4.2 list����Ԫ�ز����ɾ������
push_back(elem);//������β������һ��Ԫ��
pop_back();//ɾ�����������һ��Ԫ��
push_front(elem);//��������ͷ����һ��Ԫ��
pop_front();//��������ͷ�Ƴ���һ��Ԫ��
insert(pos,elem);//��posλ�ò�elemԪ�صĿ��������������ݵ�λ�á�
insert(pos,n,elem);//��posλ�ò���n��elem���ݣ��޷���ֵ��
insert(pos,beg,end);//��posλ�ò���[beg,end)��������ݣ��޷���ֵ��
clear();//�Ƴ���������������
erase(beg,end);//ɾ��[beg,end)��������ݣ�������һ�����ݵ�λ�á�
erase(pos);//ɾ��posλ�õ����ݣ�������һ�����ݵ�λ�á�
remove(elem);//ɾ��������������elemֵƥ���Ԫ�ء�
*/

void printList(const list<int>&L)
{
	for (list<int>::const_iterator it = L.begin(); it != L.end(); it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}

//void test02()
//{
//	list<int>L;
//	list<int>L2(10, 10);
//	list<int>L3(L2.begin(), L2.end());
//
//
//	list <int>L4;
//	L4.push_back(10);
//	L4.push_back(20);
//	L4.push_back(30);
//	L4.push_front(100);
//	L4.push_front(200);
//	L4.push_front(300);
//	//  300 200 100 10 20 30
//	for (list<int>::iterator it = L4.begin(); it != L4.end();it++)
//	{
//		cout << *it << " ";
//	}
//	cout << endl;
//
//	//�����ӡ
//	for (list<int>::reverse_iterator it = L4.rbegin(); it != L4.rend();it++)
//	{
//		cout << *it << " ";
//	}
//	cout << endl;
//
//	L4.insert(L4.begin(), 1000); //��������ǵ�����
//
//
//	// 1000 300 200 100 10 20 30 
//	printList(L4);
//	L4.push_back(300);
//	// 1000 300 200 100 10 20 30 300
//
//	//remove(elem);//ɾ��������������elemֵƥ���Ԫ�ء�
//	L4.remove(300);
//	printList(L4);
//
//
//}

/*
3.6.4.3 list��С����
size();//����������Ԫ�صĸ���
empty();//�ж������Ƿ�Ϊ��
resize(num);//����ָ�������ĳ���Ϊnum��
�������䳤������Ĭ��ֵ�����λ�á�
���������̣���ĩβ�����������ȵ�Ԫ�ر�ɾ����
resize(num, elem);//����ָ�������ĳ���Ϊnum��
�������䳤������elemֵ�����λ�á�
���������̣���ĩβ�����������ȵ�Ԫ�ر�ɾ����

3.6.4.4 list��ֵ����
assign(beg, end);//��[beg, end)�����е����ݿ�����ֵ������
assign(n, elem);//��n��elem������ֵ������
list& operator=(const list &lst);//���صȺŲ�����
swap(lst);//��lst�뱾���Ԫ�ػ�����
3.6.4.5 list���ݵĴ�ȡ
front();//���ص�һ��Ԫ�ء�
back();//�������һ��Ԫ�ء�
*/

/*
void test03()
{
	list <int>L;
	L.push_back(10);
	L.push_back(20);
	L.push_back(30);
	L.push_front(100);
	L.push_front(200);
	L.push_front(300);



	list <int>L2;
	L2.assign(10, 100);
	printList(L2);


	L2.assign(L.begin(), L.end());
	printList(L2);


	cout << "L2 front = " << L2.front() << endl;
	cout << "L2 back = " << L2.back() << endl;

}


/*
3.6.4.6 list��ת����
reverse();//��ת��������lst����1,3,5Ԫ�أ����д˷�����lst�Ͱ���5,3,1Ԫ�ء�
sort(); //list����
*/
/*
bool myCompare(int v1, int v2)
{
	return v1 > v2;
}

void test04()
{
	list <int>L;
	L.push_back(10);
	L.push_back(20);
	L.push_back(30);
	L.push_front(100);
	L.push_front(200);
	L.push_front(300);

	//��ת  �ʱ��㷨
	L.reverse();

	printList(L);

	//����  
	// ����ϵͳ�ṩ��׼�㷨  ʹ�õ������ṩ�ĵ���������֧���������
	// ��֧��������ʵĵ����������� ���ڲ����Ӧ�ṩ��Ӧ���㷨�Ľӿ�
	//sort(L.begin(), L.end());
	L.sort(); //Ĭ���������  ��С����

	//�޸�������� Ϊ �Ӵ�С
	L.sort(myCompare);

	printList(L);
}
*/
class Person
{
public:
	Person(string name, int age ,int height)
	{
		this->m_Name = name;
		this->m_Age = age;
		this->m_Height = height;
	}

	bool operator==(const Person &p)
	{
		if (this->m_Name == p.m_Name && this->m_Age == p.m_Age && this->m_Height == p.m_Height)
		{
			return true;
		}
		return false;
	
	}

	string m_Name;
	int m_Age;
	int m_Height; //���
};

bool myComparePerson(Person & p1, Person &p2)
{
	//��������  ����
	// ���������ͬ ������� ���н���

	if (p1.m_Age == p2.m_Age)
	{
		return p1.m_Height > p2.m_Height;
	}

	return p1.m_Age < p2.m_Age;
}

void test05()
{
	list<Person> L;

	Person p1("����", 30 , 170);
	Person p2("����", 28 , 160);
	Person p3("����", 24 , 150);
	Person p4("����", 24 , 166);
	Person p5("����", 24 , 158);
	Person p6("үү", 90 , 200);
	Person p7("�߾�", 999 , 999);

	L.push_back(p1);
	L.push_back(p2);
	L.push_back(p3);
	L.push_back(p4);
	L.push_back(p5);
	L.push_back(p6);
	L.push_back(p7);

	for (list<Person>::iterator it = L.begin(); it != L.end();it++)
	{
		cout << " ������ " << it->m_Name << " ���䣺 " << it->m_Age <<" ��ߣ� "<< it->m_Height <<  endl;
	}
	cout << "�����Ľ��Ϊ�� " << endl;
	L.sort(myComparePerson); //�Զ������������ ����ָ���������
	for (list<Person>::iterator it = L.begin(); it != L.end(); it++)
	{
		cout << " ������ " << it->m_Name << " ���䣺 " << it->m_Age << " ��ߣ� " << it->m_Height << endl;
	}

	//L.remove(p1);

	cout << "ɾ�����޺�Ľ��Ϊ�� " << endl;
	for (list<Person>::iterator it = L.begin(); it != L.end(); it++)
	{
		cout << " ������ " << it->m_Name << " ���䣺 " << it->m_Age << " ��ߣ� " << it->m_Height << endl;
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