#include<iostream>
#include<string>
using namespace std;

// Ҫʵ����Ԫ������Ϊģ�沢��������ʵ�֣�����2������Ҫ�Ĳ���
// ��һ�������߱�����Person�࣬�Ȳ�����
template<class T1, class T2> class Person;
// �ڶ��������߱���������һ������ģ������
template<class T1, class T2> void printPerson2(Person<T1, T2>& p);

// ��ģ���������ʵ�ַ���һ��
template<class T1, class T2> void printPerson3(Person<T1, T2>& p) {
	cout << "����ʵ��2�������� " << p.m_Name << ", ���䣺 " << p.m_Age << endl;
}

template<class T1, class T2>
class Person {
	// 1.ȫ�ֺ��������Ԫ��������ʵ��
	friend void printPerson(Person<T1, T2>& p) {
		cout << "����ʵ�֣������� " << p.m_Name << ", ���䣺 " << p.m_Age << endl;
	}
	// 2.ȫ�ֺ��������Ԫ������ʵ��
// ������: ����ģ�溯��Ϊ��Ԫ,����Ҫ�ռ�<>������ģ��
	friend void printPerson2<>(Person<T1, T2>& p);

	// 3. ȫ�ֺ��������Ԫ������ʵ��,��ģ��ʵ�ֺ���������һ��
	friend void printPerson3<>(Person<T1, T2>& p);
public:
	Person(T1 name, T2 age) {
		this->m_Age = age;
		this->m_Name = age;
	}
private:
	T1 m_Name;
	T2 m_Age;
};

// ���Ĳ�������ʵ��
template<class T1, class T2>
void printPerson2(Person<T1, T2>& p) {
	cout << "����ʵ��1�������� " << p.m_Name << ", ���䣺 " << p.m_Age << endl;
}

void test01() {
	Person<string, int> p("Tom", 29);

	printPerson(p);
	printPerson2(p);
	printPerson3(p);

}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}