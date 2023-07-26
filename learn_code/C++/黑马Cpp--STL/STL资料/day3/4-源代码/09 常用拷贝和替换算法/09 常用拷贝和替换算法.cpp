#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <algorithm>
#include <iterator>
/*
copy算法 将容器内指定范围的元素拷贝到另一容器中
@param beg 容器开始迭代器
@param end 容器结束迭代器
@param dest 目标起始迭代器
*/
void test01()
{
	vector<int>v1;

	for (int i = 0; i < 10;i++)
	{
		v1.push_back(i);
	}

	vector<int>vTarget;
	vTarget.resize(v1.size());

	copy(v1.begin(), v1.end(), vTarget.begin());

	//for_each(vTarget.begin(), vTarget.end(), [](int val){ cout << val << " "; });

	copy(vTarget.begin(), vTarget.end(), ostream_iterator<int>(cout, " "));

	cout << endl;

}

/*
replace算法 将容器内指定范围的旧元素修改为新元素
@param beg 容器开始迭代器
@param end 容器结束迭代器
@param oldvalue 旧元素
@param oldvalue 新元素

replace_if算法 将容器内指定范围满足条件的元素替换为新元素
@param beg 容器开始迭代器
@param end 容器结束迭代器
@param callback函数回调或者谓词(返回Bool类型的函数对象)
@param oldvalue 新元素
*/

class MyCompare2
{
public:
	bool operator()(int val)
	{
		return val > 3;
	}
};
void test02()
{
	vector<int>v1;

	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
	}

	replace(v1.begin(), v1.end(), 3, 300);

	copy(v1.begin(), v1.end(), ostream_iterator<int>(cout, " "));
	cout << endl;

	//按条件 进行替换   将所有大于3  替换为 3000
	replace_if(v1.begin(), v1.end(), MyCompare2(), 3000);

	copy(v1.begin(), v1.end(), ostream_iterator<int>(cout, " "));

}


/*
swap算法 互换两个容器的元素
@param c1容器1
@param c2容器2
*/

void test03()
{
	vector<int>v1;

	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
	}


	vector<int>v2(10, 100);

	swap(v1, v2);

	copy(v1.begin(), v1.end(), ostream_iterator<int>(cout, " "));
	cout << endl;


}

int main(){
	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;
}