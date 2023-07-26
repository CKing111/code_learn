#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;
#include <deque>
#include <algorithm>

/*
3.3.3.1 deque���캯��
deque<T> deqT;//Ĭ�Ϲ�����ʽ
deque(beg, end);//���캯����[beg, end)�����е�Ԫ�ؿ���������
deque(n, elem);//���캯����n��elem����������
deque(const deque &deq);//�������캯����

3.3.3.2 deque��ֵ����
assign(beg, end);//��[beg, end)�����е����ݿ�����ֵ������
assign(n, elem);//��n��elem������ֵ������
deque& operator=(const deque &deq); //���صȺŲ�����
swap(deq);// ��deq�뱾���Ԫ�ػ���

3.3.3.3 deque��С����
deque.size();//����������Ԫ�صĸ���
deque.empty();//�ж������Ƿ�Ϊ��
deque.resize(num);//����ָ�������ĳ���Ϊnum,�������䳤������Ĭ��ֵ�����λ�á����������̣���ĩβ�����������ȵ�Ԫ�ر�ɾ����
deque.resize(num, elem); //����ָ�������ĳ���Ϊnum,�������䳤������elemֵ�����λ��,���������̣���ĩβ�����������ȵ�Ԫ�ر�ɾ����
*/

void printDeque(const deque<int>&d)
{

	// iterator ��ͨ������
	// reverse_iterator ��ת������
	// const_iterator   ֻ��������
	for (deque<int>::const_iterator it = d.begin(); it != d.end();it++)
	{
		//*it = 10000;
		cout << *it << " ";
	}
	cout << endl;
}

void test01()
{
	deque<int>d;
	d.push_back(10);
	d.push_back(30);
	d.push_back(20);
	d.push_back(40);

	printDeque(d);

	deque<int>d2(10, 10);

	d.swap(d2);

	printDeque(d);

	if (d.empty())
	{
		cout << "dΪ��" << endl;
	}
	else
	{
		cout << "d��Ϊ��--size = " << d.size() << endl;
	}

}

/*
3.3.3.4 deque˫�˲����ɾ������
push_back(elem);//������β�����һ������
push_front(elem);//������ͷ������һ������
pop_back();//ɾ���������һ������
pop_front();//ɾ��������һ������

3.3.3.5 deque���ݴ�ȡ
at(idx);//��������idx��ָ�����ݣ����idxԽ�磬�׳�out_of_range��
operator[];//��������idx��ָ�����ݣ����idxԽ�磬���׳��쳣��ֱ�ӳ���
front();//���ص�һ�����ݡ�
back();//�������һ������
3.3.3.6 deque�������
insert(pos,elem);//��posλ�ò���һ��elemԪ�صĿ��������������ݵ�λ�á�
insert(pos,n,elem);//��posλ�ò���n��elem���ݣ��޷���ֵ��
insert(pos,beg,end);//��posλ�ò���[beg,end)��������ݣ��޷���ֵ��
3.3.3.7 dequeɾ������
clear();//�Ƴ���������������
erase(beg,end);//ɾ��[beg,end)��������ݣ�������һ�����ݵ�λ�á�
erase(pos);//ɾ��posλ�õ����ݣ�������һ�����ݵ�λ�á�
*/

void test02()
{
	deque<int>d;
	d.push_back(10);
	d.push_back(20);
	d.push_back(30);
	d.push_back(40);
	d.push_front(100);
	d.push_front(200);
	d.push_front(300);
	d.push_front(400);

	printDeque(d); //   400 300 200 100 10 20 30 40

	d.pop_back(); //ɾ��  40
	d.pop_front(); // ɾ�� 400

	printDeque(d); //    300 200 100 10 20 30 


	cout << "��һ��Ԫ�أ� " << d.front() << endl;


	cout << "���һ��Ԫ�أ� " << d.back() << endl;

	//���� 

	d.insert(++d.begin(), 10000);

	cout << "d.begin()" << &(d.begin()) << endl;
	cout << "++d.begin()" << &(++d.begin())<< endl;
	//cout << "1+d.begin()" << &(1+d.begin()) << endl;
	//cout << "4+d.begin()" << &(4 + d.begin()) << endl;
	cout << "++d.begin() - d.begin()" << &(1 + d.begin())  - &(d.begin()) << endl;

	printDeque(d);  //    300  10000 200 100 10 20 30 


	//ɾ�� 
	d.erase(++d.begin(),--d.end()); //ɾ������  10000 �� 20�����䶼ɾ����

	printDeque(d);


}

bool myCompare(int v1 ,int v2)
{
	return v1 > v2; //����
}

void test03()
{
	//����sort����
	deque<int>d;
	d.push_back(10);
	d.push_back(20);
	d.push_back(30);
	d.push_back(40);
	d.push_front(100);
	d.push_front(200);
	d.push_front(300);
	d.push_front(400);

	//Ĭ����������С����
	sort(d.begin(), d.end());

	//�Ӵ�С����
	sort(d.begin(), d.end(), myCompare);

	printDeque(d);
}

int main(){
	//test01();
	//test02();
	test03();

	system("pause");
	return EXIT_SUCCESS;
}