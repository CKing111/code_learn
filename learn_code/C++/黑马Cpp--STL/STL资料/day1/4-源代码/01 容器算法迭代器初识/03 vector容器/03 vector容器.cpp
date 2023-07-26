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
		cout << v.capacity() << endl;  // v.capacity()����������
	}
}



/*
3.2.4.1 vector���캯��
vector<T> v; //����ģ��ʵ����ʵ�֣�Ĭ�Ϲ��캯��
vector(v.begin(), v.end());//��v[begin(), end())�����е�Ԫ�ؿ���������
vector(n, elem);//���캯����n��elem����������
vector(const vector &vec);//�������캯����

//���� ʹ�õڶ������캯�� ���ǿ���...
int arr[] = {2,3,4,1,9};
vector<int> v1(arr, arr + sizeof(arr) / sizeof(int));

3.2.4.2 vector���ø�ֵ����
assign(beg, end);//��[beg, end)�����е����ݿ�����ֵ������
assign(n, elem);//��n��elem������ֵ������
vector& operator=(const vector  &vec);//���صȺŲ�����
swap(vec);// ��vec�뱾���Ԫ�ػ�����
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
	//����
	vector<int>v1;

	vector<int>v2(10, 100);
	printVector(v2);

	vector<int>v3(v2.begin(), v2.end());
	printVector(v3);


	//��ֵ
	vector<int>v4;
	//v4.assign(v3.begin(), v3.end());
	v4 = v3;
	printVector(v4);

	int arr[] = { 2, 3, 4, 1, 9 };
	vector<int> v5(arr, arr + sizeof(arr) / sizeof(int));


	//swap����
	v4.swap(v5);
	printVector(v4);

}


/*
3.2.4.3 vector��С����
size();//����������Ԫ�صĸ���
empty();//�ж������Ƿ�Ϊ��
resize(int num);//����ָ�������ĳ���Ϊnum���������䳤������Ĭ��ֵ�����λ�á����������̣���ĩβ�����������ȵ�Ԫ�ر�ɾ����
resize(int num, elem);//����ָ�������ĳ���Ϊnum���������䳤������elemֵ�����λ�á����������̣���ĩβ����������>�ȵ�Ԫ�ر�ɾ����
capacity();//����������
reserve(int len);//����Ԥ��len��Ԫ�س��ȣ�Ԥ��λ�ò���ʼ����Ԫ�ز��ɷ��ʡ�

3.2.4.4 vector���ݴ�ȡ����
at(int idx); //��������idx��ָ�����ݣ����idxԽ�磬�׳�out_of_range�쳣��
operator[];//��������idx��ָ�����ݣ�Խ��ʱ������ֱ�ӱ���
front();//���������е�һ������Ԫ��
back();//�������������һ������Ԫ��

3.2.4.5 vector�����ɾ������
insert(const_iterator pos, int count,ele);//������ָ��λ��pos����count��Ԫ��ele.
push_back(ele); //β������Ԫ��ele
pop_back();//ɾ�����һ��Ԫ��
erase(const_iterator start, const_iterator end);//ɾ����������start��end֮���Ԫ��
erase(const_iterator pos);//ɾ��������ָ���Ԫ��
clear();//ɾ������������Ԫ��
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
		cout << "v1Ϊ��" << endl;
	}
	else
	{
		cout << "v1��Ϊ��" << endl;
	}

	//����ָ����������  resize
	v1.resize(10,1000); //�ڶ���������Ĭ������ֵ�������дĬ��ֵΪ0

	printVector(v1);

	v1.resize(3);

	printVector(v1);


	cout << "v1�ĵ�һ��Ԫ�أ� " << v1.front() << endl;

	cout << "v1�����һ��Ԫ�أ� " << v1.back() << endl;

	v1.insert(v1.begin(), 2,1000); //����1 �ǵ�����
	//  1000  1000  10  40  20
	printVector(v1);

	v1.pop_back(); //βɾ
	//  1000  1000  10  40  
	printVector(v1);

	//ɾ��
	//v1.erase(v1.begin() , v1.end());
	//���
	v1.clear();
	printVector(v1);

}


//����swap�����ڴ�
void test04()
{
	vector<int>v;
	for (int i = 0; i < 100000;i++)
	{
		v.push_back(i);
	}
	cout << "v�������� " << v.capacity() << endl;
	cout << "v�Ĵ�С�� " << v.size() << endl;

	v.resize(3);

	cout << "v�������� " << v.capacity() << endl;
	cout << "v�Ĵ�С�� " << v.size() << endl;

	//�����ڴ�
	vector<int>(v).swap(v);
	cout << "v�������� " << v.capacity() << endl;
	cout << "v�Ĵ�С�� " << v.size() << endl;

}


//����reverseԤ���ռ�
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
	//�������
	vector<int>v1;
	v1.push_back(10);
	v1.push_back(40);
	v1.push_back(20);
	v1.push_back(30);
	cout << "������������ " << endl;
	printVector(v1);

	cout << "������������ " << endl;

	for (vector<int>::reverse_iterator it = v1.rbegin(); it != v1.rend();it++)
	{
		cout << *it << endl;
	}


	// vector�����ĵ�����  ������ʵ�����
	//����ж�һ�������ĵ������Ƿ�֧���������

	vector<int>::iterator itBegin = v1.begin();

	itBegin = itBegin + 2; //����﷨ͨ�� ֧���������



	list<int>L;
	L.push_back(10);
	L.push_back(20);
	L.push_back(30);

	list<int>::iterator it2 = L.begin();
	//it2 = it2+1; //list�����ĵ�������֧���������
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