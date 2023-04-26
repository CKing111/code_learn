# include<iostream>
# include<string>
using namespace std;

// ��ģ��
// 1. template��������ŵ��������࣬���г�Ա���ͺͺ���ΪT
// 2. ��ģ�����Ϳ�����Ĭ�ϲ���


// ʲô�Ƿ��ͱ�̣����Ͳ�����


template<class NAMETYPE, class AGETYPE = int>
class Person {
public:
	Person(NAMETYPE name, AGETYPE age) {
		this->m_Age = age;
		this->m_Name = name;
	}


	NAMETYPE m_Name;
	AGETYPE m_Age;
};


void test01() {
	//Person p1("Tom", 19);		// ʧ�ܣ�������ģ�治����ʹ���Զ������Ƶ�
	Person<string, int> p1("Tom", 19);	// ��Ҫ��ʾָ������
	cout << "��׼��ģ�棬������" << p1.m_Name << ", ���䣺" << p1.m_Age << endl;
	
	Person<string> p2("Jerry", 20);	// ��Ҫ��ʾָ������
	cout << "Ĭ�ϲ�����ģ�棬������" << p2.m_Name << ", ���䣺" << p2.m_Age << endl;

}


int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}