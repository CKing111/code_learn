#include<iostream>
#include<vector> // vector容器的头文件
#include<algorithm>	 // 迭代算法头文件，for_each
#include<string>


using namespace std;

// 普通指针也属于一种迭代器
void test01() 
{
	int arr[5] = { 1,2,3,4,5 };
	int *p = arr;					 // 初始化为指向数组 arr 的首元素的地址
	for (int i = 0; i < 5; i++)
	{
		//cout << arr[i] << endl;	 // 数组遍历
		cout << *(p++) << endl;      // 指针遍历， *(p++) 表示先取出指针 p 所指向的元素的值，再将指针 p 指向下一个元素的地址
		// 这个过程中，指针 p 的值发生了变化，指向了数组中的下一个元素，从而实现了遍历整个数组的目的。
	}
}


// for_each的迭代器：
void myPrint(int val) {
	cout << val << endl;
}
// 使用内置属性类型vector  需要#include<vector>
void test02()
{
	vector<int>v; // 声明容器

	// 尾插法插入数据
	v.push_back(10);
	v.push_back(20);
	v.push_back(30);
	v.push_back(40);
	v.push_back(50);

	// 通过迭代器可以遍历容器
	// 每个容器有自己专属的迭代器
	vector<int>::iterator itBegin = v.begin();  // 起始迭代器
	vector<int>::iterator itEnd = v.end();		// 结束迭代器，指向最后一个元素的下一个地址，不可以解引用

	// 遍历
	// 第一种方法，复杂，需要明确开始结束迭代器未知
	while (itBegin != itEnd)
	{
		cout << *itBegin << endl;
		itBegin++;
	}

	// 方法2：简单
	for (vector<int>::iterator it = v.begin(); it != v.end(); it++)
	{
		cout << *it << endl;
	}			 

	// 方法3：利用系统提供的算法，需要include头文件<algorithm>
	for_each(v.begin(), v.end(), myPrint); // 参数：起始迭代器，结束迭代器，回调函数
	/*
	系统实际操作
	void _For_each_unchecked(_InIt _First, _InIt _Last, _Fn1& _Func)
	{	// perform function for each element
	for (; _First != _Last; ++_First)
		_Func(*_First);
	}
	*/

}

// 自定义的数据类型
class Person
{
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}

//private:
	string m_Name;
	int m_Age;
};
void test03()
{
	vector<Person> v; // 声明容器，存放Person类型数据

	// 实例化对象
	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);
	Person p5("eee", 50);
	// 输入容器
	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);
	v.push_back(p5);

	// 遍历
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++)
	{
		// *it是借引用的Person类型；
		// it是Person指针；
		cout << "姓名：" << (*it).m_Name << "，年龄：" << it->m_Age << endl;
	}
}

// 存放自定义数据类型的指针
void test04()
{
	vector<Person*> v;
	// 实例化对象
	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);
	Person p5("eee", 50);
	// 存入Person类型的引用
	// 输入容器
	v.push_back(&p1);
	v.push_back(&p2);
	v.push_back(&p3);
	v.push_back(&p4);
	v.push_back(&p5);

	// 遍历
	for (vector<Person*>::iterator it = v.begin(); it != v.end(); it++)
	{
		// it是Person的引用，*it是解引用的Person类型
		cout << "姓名： " << (*it)->m_Name << ", 年龄：" << (*it)->m_Age << endl;
	}
}


// 容器嵌套容器
void test05()
{
	vector<vector<int>> v;  // 类似二维数组

	// 初始化第一维的容器
	vector<int>v1;
	vector<int>v2;
	vector<int>v3;
	// 填充数据
	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
		v2.push_back(i + 10);
		v3.push_back(i + 100);
	}
	// 将一维vector插入大容器
	v.push_back(v1);
	v.push_back(v2);
	v.push_back(v3);
	
	//遍历
	int count = 0;
	for (vector<vector<int>>::iterator it = v.begin(); it != v.end(); it++)		 // 遍历第一层数据
	{
		// *it是vector<int>
		
		cout << "第一位都第" << count + 1 << "个vector容器：" << endl;
		for (vector<int>::iterator vit = (*it).begin(); vit != (*it).end(); vit++)
		{
			// *vit ---int值
			cout << *vit << " ";
		}
		cout << endl;
		count++;

	}

}



// sort算法的使用
// 自定义的数据类型
class Person2
{
public:
	Person2(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}

	string getName() { return m_Name; }
	int getAge() { return m_Age; }
private:
	string m_Name;
	int m_Age;
};
// 比较函数
//bool compare_by_name( Person2 a,  Person2 b) {
//	return a.getName() < b.getName;
//}

bool compare_by_age( Person2 a,  Person2 b) {
	return a.getAge() < b.getAge();
}
void test06()
{
	vector<Person2> v; // 声明容器，存放Person类型数据

					  // 实例化对象
	Person2 p1("aaa", 18);
	Person2 p2("dsa", 23);
	Person2 p3("ccc", 16);
	Person2 p4("bbb", 20);
	Person2 p5("eee", 19);
	// 输入容器
	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);
	v.push_back(p5);


	// 排序
	sort(v.begin(), v.end(), compare_by_age);	// 年龄排序
	// 遍历
	cout << "按照年龄排序：" << endl;
	for (vector<Person2>::iterator it = v.begin(); it != v.end(); it++)
	{
		// *it是借引用的Person类型；
		// it是Person指针；
		cout << "姓名：" << (*it).getName() << "，年龄：" << it->getAge() << endl;
	}
	//sort(v.begin(), v.end(), compare_by_name);	 // 姓名排序
	cout << "按照年龄排序：" << endl;
	for (vector<Person2>::iterator it = v.begin(); it != v.end(); it++)
	{
		// *it是借引用的Person类型；  
		// it是Person指针；
		cout << "姓名：" << (*it).getName() << "，年龄：" << it->getAge() << endl;
	}

}

int main() {
	//test01();
	//test02(); 
	//test03();
	//test04();
	//test05();

	test06();  //**
	system("pause");
	return EXIT_SUCCESS;
}