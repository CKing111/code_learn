#include <iostream>;
#include<string>


using namespace std;


class Person {
public:
	Person(string name, int age) {
		this->m_Age = age;
		this->m_Name = name;
	}

	// 重载==
	bool operator==(const Person& p) {
		/*if (this->m_Name == p.m_Name && this->m_Age == p.m_Age) {
			return true;
		}
		return false;*/
		return this->m_Name == p.m_Name && this->m_Age == p.m_Age;
	}

	bool operator!=(const Person& p) {
		return !(this->m_Name == p.m_Name && this->m_Age == p.m_Age);
	}
	string m_Name;
	int m_Age;
};

void test01() {
	Person p1("Tom", 18);
	Person p2("Tom", 19);

	if (p1 == p2) {
		cout << "p1 等于 p2" << endl;
	}
	else {
		cout << "p1 不等于 p2" << endl;
	}
	if (p1 != p2) {
		cout << "p1 不等于 p2" << endl;
	}
	else {
		cout << "p1 等于 p2" << endl;
	}
}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}