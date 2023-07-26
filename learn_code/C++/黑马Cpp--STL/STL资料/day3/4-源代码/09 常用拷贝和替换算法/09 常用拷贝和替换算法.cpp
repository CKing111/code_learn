#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <vector>
#include <algorithm>
#include <iterator>
/*
copy�㷨 ��������ָ����Χ��Ԫ�ؿ�������һ������
@param beg ������ʼ������
@param end ��������������
@param dest Ŀ����ʼ������
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
replace�㷨 ��������ָ����Χ�ľ�Ԫ���޸�Ϊ��Ԫ��
@param beg ������ʼ������
@param end ��������������
@param oldvalue ��Ԫ��
@param oldvalue ��Ԫ��

replace_if�㷨 ��������ָ����Χ����������Ԫ���滻Ϊ��Ԫ��
@param beg ������ʼ������
@param end ��������������
@param callback�����ص�����ν��(����Bool���͵ĺ�������)
@param oldvalue ��Ԫ��
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

	//������ �����滻   �����д���3  �滻Ϊ 3000
	replace_if(v1.begin(), v1.end(), MyCompare2(), 3000);

	copy(v1.begin(), v1.end(), ostream_iterator<int>(cout, " "));

}


/*
swap�㷨 ��������������Ԫ��
@param c1����1
@param c2����2
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