#include<iostream>
#include<string>

using namespace std;

// ��װclass������Ա��������Ϊprivate

class Person {
private:				// ���ⲻ�ɷ��ʣ����ڷ���
	int m_Age = 0;		// ���䣬ֻ��
	string m_Name;		// ����Ȩ�ޣ���д
	string m_Love;		// ���ˣ�ֻд

public:
	// ��������
	void setAge(int age) {
		if (age < 0 || age>100) {
			cout << "���������������" << endl;
			m_Age = age;
			return;		// ͨ����return��
		}
		m_Age = age;
	}
	// ��ȡ����
	int getAge() {
		return m_Age;
	}
	// ��ȡ����
	string getName() {
		return m_Name;
	}
	// д������
	void setName(string name) {
		m_Name = name;
	}
	// д������
	void setLove(string lover) {
		m_Love = lover;
	}
};

void test01() {
	Person p1;
	//p1.m_name;		// ���ɶ�
	p1.setName("����"); // д������
	cout << "p1������" << p1.getName() << endl;	//��ȡ����
	cout << "p1�����䣺" << p1.getAge() << endl;//��ȡ����
	p1.setAge(101);		// ��������
	cout << "p1�����䣺" << p1.getAge() << endl;//��ȡ����
	p1.setLove("����");
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}