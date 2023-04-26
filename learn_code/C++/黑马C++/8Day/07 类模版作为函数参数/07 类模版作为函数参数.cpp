# include<iostream>
# include<string>
using namespace std;

template<class NAMETYPE, class AGETYPE >
class Person {
public:
	Person(NAMETYPE name, AGETYPE age) {
		this->m_Age = age;
		this->m_Name = name;
	}


	NAMETYPE m_Name;
	AGETYPE m_Age;
};

// ��ģ�����ɵĶ�����Ϊ�������������¼��ַ�ʽ��
// 1. �ƶ���������
void doWork1(Person<string, int>& p) {
	cout << "����1���ƶ��������ͣ�������" << p.m_Name << ", ���䣺" << p.m_Age << endl;
}

// 2. ����ģ�滯
template<class T1, class T2>		// ����ģ��
void doWork2(Person<T1, T2>& p) {
	cout << "����2������ģ�滯��������" << p.m_Name << ", ���䣺" << p.m_Age << endl;
	cout << "T1 type: " << typeid(T1).name() << ", T2 type: " << typeid(T2).name() << endl;
}

// 3. ������ģ�滯
template<class T>
void doWork3(T& p) {
	cout << "����3��������ģ�滯��������" << p.m_Name << ", ���䣺" << p.m_Age << endl;
	cout << "T type: " << typeid(T).name() << endl;
}

void test01() {
	Person<string, int> p1("Tom", 29);

	// 1.
	doWork1(p1);
	doWork2(p1);
	doWork3(p1);
/*
����1���ƶ��������ͣ�������Tom, ���䣺29
����2������ģ�滯��������Tom, ���䣺29
T1 type: class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> >, T2 type: int
����3��������ģ�滯��������Tom, ���䣺29
T type: class Person<class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> >,int>
�밴���������. . .
*/
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}