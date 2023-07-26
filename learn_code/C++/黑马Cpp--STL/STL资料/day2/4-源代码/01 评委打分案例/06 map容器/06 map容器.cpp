#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
using namespace std;

#include <map> //map�� multimap ��ͷ�ļ�

/*
3.8.2.1 map���캯��
map<T1, T2> mapTT;//mapĬ�Ϲ��캯��:
map(const map &mp);//�������캯��
3.8.2.2 map��ֵ����
map& operator=(const map &mp);//���صȺŲ�����
swap(mp);//����������������
3.8.2.3 map��С����
size();//����������Ԫ�ص���Ŀ
empty();//�ж������Ƿ�Ϊ��
3.8.2.4 map��������Ԫ�ز���
map.insert(...); //����������Ԫ�أ�����pair<iterator,bool>
map<int, string> mapStu;
// ��һ�� ͨ��pair�ķ�ʽ�������
mapStu.insert(pair<int, string>(3, "С��"));
// �ڶ��� ͨ��pair�ķ�ʽ�������
mapStu.inset(make_pair(-1, "У��"));
// ������ ͨ��value_type�ķ�ʽ�������
mapStu.insert(map<int, string>::value_type(1, "С��"));
// ������ ͨ������ķ�ʽ����ֵ
mapStu[3] = "С��";
mapStu[5] = "С��";
*/

void test01()
{
	map<int, int> m;

	//���뷽ʽ
	//��һ��
	m.insert(pair<int, int>(1, 10));

	//�ڶ���
	m.insert(make_pair(2, 20));

	//������
	m.insert(map<int, int>::value_type(3, 30));

	//������
	m[4] = 40;

	for (map<int, int>::iterator it = m.begin(); it != m.end();it++)
	{
		cout << " key =  " << it->first << " value = " << (*it).second << endl;
	}


	//cout << m[4] << endl;
	//for (map<int, int>::iterator it = m.begin(); it != m.end(); it++)
	//{
	//	cout << " key =  " << it->first << " value = " << (*it).second << endl;
	//}
}

/*
3.8.2.5 mapɾ������
clear();//ɾ������Ԫ��
erase(pos);//ɾ��pos��������ָ��Ԫ�أ�������һ��Ԫ�صĵ�������
erase(beg,end);//ɾ������[beg,end)������Ԫ�� ��������һ��Ԫ�صĵ�������
erase(keyElem);//ɾ��������keyΪkeyElem�Ķ��顣
3.8.2.6 map���Ҳ���
find(key);//���Ҽ�key�Ƿ����,�����ڣ����ظü���Ԫ�صĵ�������/�������ڣ�����map.end();
count(keyElem);//����������keyΪkeyElem�Ķ����������map��˵��Ҫô��0��Ҫô��1����multimap��˵��ֵ���ܴ���1��
lower_bound(keyElem);//���ص�һ��key>=keyElemԪ�صĵ�������
upper_bound(keyElem);//���ص�һ��key>keyElemԪ�صĵ�������
equal_range(keyElem);//����������key��keyElem��ȵ������޵�������������
*/
void test02()
{
	map<int, int> m;
	m.insert(pair<int, int>(1, 10));
	m.insert(make_pair(2, 20));
	m.insert(map<int, int>::value_type(3, 30));
	m[4] = 40;

	//m.erase(3); //����key����ɾ��Ԫ��
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++)
	{
		cout << " key =  " << it->first << " value = " << (*it).second << endl;
	}

	//����
	map<int,int>::iterator pos =  m.find(3);
	if (pos != m.end())
	{
		cout << "�ҵ��� keyΪ�� " << (*pos).first << " value Ϊ�� " << pos->second << endl;
	}

	int num  = m.count(4);
	cout << "keyΪ4�Ķ������Ϊ�� " << num << endl;

	//lower_bound(keyElem);//���ص�һ��key>=keyElemԪ�صĵ�������
	map<int,int>::iterator ret =  m.lower_bound(3);
	if (ret != m.end())
	{
		cout << "�ҵ���lower_bound��keyΪ��  " << ret->first << " value =  " << ret->second << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}
	//upper_bound(keyElem);//���ص�һ��key>keyElemԪ�صĵ�������
	ret=  m.upper_bound(3);
	if (ret != m.end())
	{
		cout << "�ҵ���upper_bound��keyΪ��  " << ret->first << " value =  " << ret->second << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}

	//equal_range(keyElem);//����������key��keyElem��ȵ������޵�������������

	pair< map<int, int>::iterator, map<int, int>::iterator> it2 = m.equal_range(3);

	if ( it2.first != m.end())
	{
		cout << "�ҵ���equal_range�е� lower_bound��keyΪ��  " << it2.first->first << " value =  " << it2.first->second << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}

	if (it2.second != m.end())
	{
		cout << "�ҵ���equal_range�е� upper_bound��keyΪ��  " << it2.second->first << " value =  " << it2.second->second << endl;
	}
	else
	{
		cout << "δ�ҵ�" << endl;
	}
}

class MyCompare
{
public:
	bool operator()(int v1,int v2)
	{
		return v1 > v2;
	}

};

//ָ��map�������������
void test03()
{
	map<int, int, MyCompare> m;
	m.insert(pair<int, int>(1, 10));
	m.insert(make_pair(2, 20));
	m.insert(map<int, int>::value_type(3, 30));
	m[4] = 40;

	for (map<int, int, MyCompare>::iterator it = m.begin(); it != m.end(); it++)
	{
		cout << " key =  " << it->first << " value = " << (*it).second << endl;
	}

}


int main(){
	//test01();
	//test02();
	test03();

	system("pause");
	return EXIT_SUCCESS;
}