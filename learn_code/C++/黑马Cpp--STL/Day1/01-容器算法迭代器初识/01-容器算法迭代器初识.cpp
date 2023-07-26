#include<iostream>
#include<vector> // vector������ͷ�ļ�
#include<algorithm>	 // �����㷨ͷ�ļ���for_each
#include<string>


using namespace std;

// ��ָͨ��Ҳ����һ�ֵ�����
void test01() 
{
	int arr[5] = { 1,2,3,4,5 };
	int *p = arr;					 // ��ʼ��Ϊָ������ arr ����Ԫ�صĵ�ַ
	for (int i = 0; i < 5; i++)
	{
		//cout << arr[i] << endl;	 // �������
		cout << *(p++) << endl;      // ָ������� *(p++) ��ʾ��ȡ��ָ�� p ��ָ���Ԫ�ص�ֵ���ٽ�ָ�� p ָ����һ��Ԫ�صĵ�ַ
		// ��������У�ָ�� p ��ֵ�����˱仯��ָ���������е���һ��Ԫ�أ��Ӷ�ʵ���˱������������Ŀ�ġ�
	}
}


// for_each�ĵ�������
void myPrint(int val) {
	cout << val << endl;
}
// ʹ��������������vector  ��Ҫ#include<vector>
void test02()
{
	vector<int>v; // ��������

	// β�巨��������
	v.push_back(10);
	v.push_back(20);
	v.push_back(30);
	v.push_back(40);
	v.push_back(50);

	// ͨ�����������Ա�������
	// ÿ���������Լ�ר���ĵ�����
	vector<int>::iterator itBegin = v.begin();  // ��ʼ������
	vector<int>::iterator itEnd = v.end();		// ������������ָ�����һ��Ԫ�ص���һ����ַ�������Խ�����

	// ����
	// ��һ�ַ��������ӣ���Ҫ��ȷ��ʼ����������δ֪
	while (itBegin != itEnd)
	{
		cout << *itBegin << endl;
		itBegin++;
	}

	// ����2����
	for (vector<int>::iterator it = v.begin(); it != v.end(); it++)
	{
		cout << *it << endl;
	}			 

	// ����3������ϵͳ�ṩ���㷨����Ҫincludeͷ�ļ�<algorithm>
	for_each(v.begin(), v.end(), myPrint); // ��������ʼ���������������������ص�����
	/*
	ϵͳʵ�ʲ���
	void _For_each_unchecked(_InIt _First, _InIt _Last, _Fn1& _Func)
	{	// perform function for each element
	for (; _First != _Last; ++_First)
		_Func(*_First);
	}
	*/

}

// �Զ������������
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
	vector<Person> v; // �������������Person��������

	// ʵ��������
	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);
	Person p5("eee", 50);
	// ��������
	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);
	v.push_back(p5);

	// ����
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++)
	{
		// *it�ǽ����õ�Person���ͣ�
		// it��Personָ�룻
		cout << "������" << (*it).m_Name << "�����䣺" << it->m_Age << endl;
	}
}

// ����Զ����������͵�ָ��
void test04()
{
	vector<Person*> v;
	// ʵ��������
	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);
	Person p5("eee", 50);
	// ����Person���͵�����
	// ��������
	v.push_back(&p1);
	v.push_back(&p2);
	v.push_back(&p3);
	v.push_back(&p4);
	v.push_back(&p5);

	// ����
	for (vector<Person*>::iterator it = v.begin(); it != v.end(); it++)
	{
		// it��Person�����ã�*it�ǽ����õ�Person����
		cout << "������ " << (*it)->m_Name << ", ���䣺" << (*it)->m_Age << endl;
	}
}


// ����Ƕ������
void test05()
{
	vector<vector<int>> v;  // ���ƶ�ά����

	// ��ʼ����һά������
	vector<int>v1;
	vector<int>v2;
	vector<int>v3;
	// �������
	for (int i = 0; i < 10; i++)
	{
		v1.push_back(i);
		v2.push_back(i + 10);
		v3.push_back(i + 100);
	}
	// ��һάvector���������
	v.push_back(v1);
	v.push_back(v2);
	v.push_back(v3);
	
	//����
	int count = 0;
	for (vector<vector<int>>::iterator it = v.begin(); it != v.end(); it++)		 // ������һ������
	{
		// *it��vector<int>
		
		cout << "��һλ����" << count + 1 << "��vector������" << endl;
		for (vector<int>::iterator vit = (*it).begin(); vit != (*it).end(); vit++)
		{
			// *vit ---intֵ
			cout << *vit << " ";
		}
		cout << endl;
		count++;

	}

}



// sort�㷨��ʹ��
// �Զ������������
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
// �ȽϺ���
//bool compare_by_name( Person2 a,  Person2 b) {
//	return a.getName() < b.getName;
//}

bool compare_by_age( Person2 a,  Person2 b) {
	return a.getAge() < b.getAge();
}
void test06()
{
	vector<Person2> v; // �������������Person��������

					  // ʵ��������
	Person2 p1("aaa", 18);
	Person2 p2("dsa", 23);
	Person2 p3("ccc", 16);
	Person2 p4("bbb", 20);
	Person2 p5("eee", 19);
	// ��������
	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);
	v.push_back(p5);


	// ����
	sort(v.begin(), v.end(), compare_by_age);	// ��������
	// ����
	cout << "������������" << endl;
	for (vector<Person2>::iterator it = v.begin(); it != v.end(); it++)
	{
		// *it�ǽ����õ�Person���ͣ�
		// it��Personָ�룻
		cout << "������" << (*it).getName() << "�����䣺" << it->getAge() << endl;
	}
	//sort(v.begin(), v.end(), compare_by_name);	 // ��������
	cout << "������������" << endl;
	for (vector<Person2>::iterator it = v.begin(); it != v.end(); it++)
	{
		// *it�ǽ����õ�Person���ͣ�  
		// it��Personָ�룻
		cout << "������" << (*it).getName() << "�����䣺" << it->getAge() << endl;
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