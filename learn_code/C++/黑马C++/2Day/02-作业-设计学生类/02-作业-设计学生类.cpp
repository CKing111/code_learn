#include<iostream>
#include<string>
using namespace std;

/*
	��Ŀ���ܣ�
		2.���һ��ѧ���࣬������������ѧ�ţ����Ը�������ѧ�Ÿ�ֵ��������ʾѧ����������ѧ��
	��Ŀ˼·��
		
*/

class Student {
public:
// ��Ա����
	string m_Name;
	int m_Id;

// ��Ա����
	// ��������
	void setName(string name) {
		m_Name = name;
	}
	// ����ѧ��
	void setId(int id) {
		m_Id = id;
	}
	// ��ӡ��Ϣ
	void showInfo() {
		cout <<"ѧ������Ϊ�� " << m_Name << ", ѧ��ѧ��Ϊ�� " << m_Id << endl;
	}
};

void test01() {
	// ����һ��ѧ����ʵ����--ͨ��һ��������������Ĺ���
	Student s1;
	s1.setName("����");
	s1.setId(1);

	// ͨ��s1���Դ�ӡ��Ϣ
	cout << "s1������Ϊ�� " << s1.m_Name << ", s1��ѧ�ţ� " << s1.m_Id << endl;
	
	// ͨ����Ա������ӡs1��Ϣ
	s1.showInfo();
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}