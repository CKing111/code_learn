#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <algorithm>
#include<list>
/*
list是一个双向循环链表
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
3.6.4.1 list构造函数
list<T> lstT;//list采用采用模板类实现,对象的默认构造形式：
list(beg,end);//构造函数将[beg, end)区间中的元素拷贝给本身。
list(n,elem);//构造函数将n个elem拷贝给本身。
list(const list &lst);//拷贝构造函数。
3.6.4.2 list数据元素插入和删除操作
push_back(elem);//在容器尾部加入一个元素
pop_back();//删除容器中最后一个元素
push_front(elem);//在容器开头插入一个元素
pop_front();//从容器开头移除第一个元素
insert(pos,elem);//在pos位置插elem元素的拷贝，返回新数据的位置。
insert(pos,n,elem);//在pos位置插入n个elem数据，无返回值。
insert(pos,beg,end);//在pos位置插入[beg,end)区间的数据，无返回值。
clear();//移除容器的所有数据
erase(beg,end);//删除[beg,end)区间的数据，返回下一个数据的位置。
erase(pos);//删除pos位置的数据，返回下一个数据的位置。
remove(elem);//删除容器中所有与elem值匹配的元素。
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
//	//逆序打印
//	for (list<int>::reverse_iterator it = L4.rbegin(); it != L4.rend();it++)
//	{
//		cout << *it << " ";
//	}
//	cout << endl;
//
//	L4.insert(L4.begin(), 1000); //插入参数是迭代器
//
//
//	// 1000 300 200 100 10 20 30 
//	printList(L4);
//	L4.push_back(300);
//	// 1000 300 200 100 10 20 30 300
//
//	//remove(elem);//删除容器中所有与elem值匹配的元素。
//	L4.remove(300);
//	printList(L4);
//
//
//}

/*
3.6.4.3 list大小操作
size();//返回容器中元素的个数
empty();//判断容器是否为空
resize(num);//重新指定容器的长度为num，
若容器变长，则以默认值填充新位置。
如果容器变短，则末尾超出容器长度的元素被删除。
resize(num, elem);//重新指定容器的长度为num，
若容器变长，则以elem值填充新位置。
如果容器变短，则末尾超出容器长度的元素被删除。

3.6.4.4 list赋值操作
assign(beg, end);//将[beg, end)区间中的数据拷贝赋值给本身。
assign(n, elem);//将n个elem拷贝赋值给本身。
list& operator=(const list &lst);//重载等号操作符
swap(lst);//将lst与本身的元素互换。
3.6.4.5 list数据的存取
front();//返回第一个元素。
back();//返回最后一个元素。
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
3.6.4.6 list反转排序
reverse();//反转链表，比如lst包含1,3,5元素，运行此方法后，lst就包含5,3,1元素。
sort(); //list排序
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

	//反转  质变算法
	L.reverse();

	printList(L);

	//排序  
	// 所有系统提供标准算法  使用的容器提供的迭代器必须支持随机访问
	// 不支持随机访问的迭代器的容器 ，内部会对应提供相应的算法的接口
	//sort(L.begin(), L.end());
	L.sort(); //默认排序规则  从小到大

	//修改排序规则 为 从大到小
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
	int m_Height; //身高
};

bool myComparePerson(Person & p1, Person &p2)
{
	//按照年龄  升序
	// 如果年龄相同 按照身高 进行降序

	if (p1.m_Age == p2.m_Age)
	{
		return p1.m_Height > p2.m_Height;
	}

	return p1.m_Age < p2.m_Age;
}

void test05()
{
	list<Person> L;

	Person p1("大娃", 30 , 170);
	Person p2("二娃", 28 , 160);
	Person p3("三娃", 24 , 150);
	Person p4("四娃", 24 , 166);
	Person p5("五娃", 24 , 158);
	Person p6("爷爷", 90 , 200);
	Person p7("蛇精", 999 , 999);

	L.push_back(p1);
	L.push_back(p2);
	L.push_back(p3);
	L.push_back(p4);
	L.push_back(p5);
	L.push_back(p6);
	L.push_back(p7);

	for (list<Person>::iterator it = L.begin(); it != L.end();it++)
	{
		cout << " 姓名： " << it->m_Name << " 年龄： " << it->m_Age <<" 身高： "<< it->m_Height <<  endl;
	}
	cout << "排序后的结果为： " << endl;
	L.sort(myComparePerson); //自定义的数据类型 必须指定排序规则
	for (list<Person>::iterator it = L.begin(); it != L.end(); it++)
	{
		cout << " 姓名： " << it->m_Name << " 年龄： " << it->m_Age << " 身高： " << it->m_Height << endl;
	}

	//L.remove(p1);

	cout << "删除大娃后的结果为： " << endl;
	for (list<Person>::iterator it = L.begin(); it != L.end(); it++)
	{
		cout << " 姓名： " << it->m_Name << " 年龄： " << it->m_Age << " 身高： " << it->m_Height << endl;
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