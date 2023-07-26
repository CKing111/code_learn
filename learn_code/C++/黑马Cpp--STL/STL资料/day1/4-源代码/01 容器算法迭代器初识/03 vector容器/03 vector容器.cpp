#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <list>


void test01()
{
	vector<int> v;
	for (int i = 0; i < 10; i++){
		v.push_back(i);
		cout << v.capacity() << endl;  // v.capacity()容器的容量
	}
}



/*
3.2.4.1 vector构造函数
vector<T> v; //采用模板实现类实现，默认构造函数
vector(v.begin(), v.end());//将v[begin(), end())区间中的元素拷贝给本身。
vector(n, elem);//构造函数将n个elem拷贝给本身。
vector(const vector &vec);//拷贝构造函数。

//例子 使用第二个构造函数 我们可以...
int arr[] = {2,3,4,1,9};
vector<int> v1(arr, arr + sizeof(arr) / sizeof(int));

3.2.4.2 vector常用赋值操作
assign(beg, end);//将[beg, end)区间中的数据拷贝赋值给本身。
assign(n, elem);//将n个elem拷贝赋值给本身。
vector& operator=(const vector  &vec);//重载等号操作符
swap(vec);// 将vec与本身的元素互换。
*/
void printVector(vector<int>&v)
{
	for (vector<int>::iterator it = v.begin(); it != v.end();it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}

void test02()
{
	//构造
	vector<int>v1;

	vector<int>v2(10, 100);
	printVector(v2);

	vector<int>v3(v2.begin(), v2.end());
	printVector(v3);


	//赋值
	vector<int>v4;
	//v4.assign(v3.begin(), v3.end());
	v4 = v3;
	printVector(v4);

	int arr[] = { 2, 3, 4, 1, 9 };
	vector<int> v5(arr, arr + sizeof(arr) / sizeof(int));


	//swap交换
	v4.swap(v5);
	printVector(v4);

}


/*
3.2.4.3 vector大小操作
size();//返回容器中元素的个数
empty();//判断容器是否为空
resize(int num);//重新指定容器的长度为num，若容器变长，则以默认值填充新位置。如果容器变短，则末尾超出容器长度的元素被删除。
resize(int num, elem);//重新指定容器的长度为num，若容器变长，则以elem值填充新位置。如果容器变短，则末尾超出容器长>度的元素被删除。
capacity();//容器的容量
reserve(int len);//容器预留len个元素长度，预留位置不初始化，元素不可访问。

3.2.4.4 vector数据存取操作
at(int idx); //返回索引idx所指的数据，如果idx越界，抛出out_of_range异常。
operator[];//返回索引idx所指的数据，越界时，运行直接报错
front();//返回容器中第一个数据元素
back();//返回容器中最后一个数据元素

3.2.4.5 vector插入和删除操作
insert(const_iterator pos, int count,ele);//迭代器指向位置pos插入count个元素ele.
push_back(ele); //尾部插入元素ele
pop_back();//删除最后一个元素
erase(const_iterator start, const_iterator end);//删除迭代器从start到end之间的元素
erase(const_iterator pos);//删除迭代器指向的元素
clear();//删除容器中所有元素
*/

void test03()
{
	vector<int>v1;
	v1.push_back(10);
	v1.push_back(40);
	v1.push_back(20);
	v1.push_back(30);

	cout << "size = " << v1.size() << endl;

	if (v1.empty())
	{
		cout << "v1为空" << endl;
	}
	else
	{
		cout << "v1不为空" << endl;
	}

	//重新指定容器长度  resize
	v1.resize(10,1000); //第二个参数是默认填充的值，如果不写默认值为0

	printVector(v1);

	v1.resize(3);

	printVector(v1);


	cout << "v1的第一个元素： " << v1.front() << endl;

	cout << "v1的最后一个元素： " << v1.back() << endl;

	v1.insert(v1.begin(), 2,1000); //参数1 是迭代器
	//  1000  1000  10  40  20
	printVector(v1);

	v1.pop_back(); //尾删
	//  1000  1000  10  40  
	printVector(v1);

	//删除
	//v1.erase(v1.begin() , v1.end());
	//清空
	v1.clear();
	printVector(v1);

}


//巧用swap收缩内存
void test04()
{
	vector<int>v;
	for (int i = 0; i < 100000;i++)
	{
		v.push_back(i);
	}
	cout << "v的容量： " << v.capacity() << endl;
	cout << "v的大小： " << v.size() << endl;

	v.resize(3);

	cout << "v的容量： " << v.capacity() << endl;
	cout << "v的大小： " << v.size() << endl;

	//收缩内存
	vector<int>(v).swap(v);
	cout << "v的容量： " << v.capacity() << endl;
	cout << "v的大小： " << v.size() << endl;

}


//巧用reverse预留空间
void test05()
{
	vector<int>v;

	v.reserve(100000);

	int num = 0;
	int * p = NULL;

	for (int i = 0; i < 100000; i++)
	{
		v.push_back(i);
		if (p != &v[0])
		{
			p = &v[0];
			num++;
		}
	}

	cout << "num = " << num << endl;
}


void test06()
{
	//逆序遍历
	vector<int>v1;
	v1.push_back(10);
	v1.push_back(40);
	v1.push_back(20);
	v1.push_back(30);
	cout << "正序遍历结果： " << endl;
	printVector(v1);

	cout << "逆序遍历结果： " << endl;

	for (vector<int>::reverse_iterator it = v1.rbegin(); it != v1.rend();it++)
	{
		cout << *it << endl;
	}


	// vector容器的迭代器  随机访问迭代器
	//如何判断一个容器的迭代器是否支持随机访问

	vector<int>::iterator itBegin = v1.begin();

	itBegin = itBegin + 2; //如果语法通过 支持随机访问



	list<int>L;
	L.push_back(10);
	L.push_back(20);
	L.push_back(30);

	list<int>::iterator it2 = L.begin();
	//it2 = it2+1; //list容器的迭代器不支持随机访问
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