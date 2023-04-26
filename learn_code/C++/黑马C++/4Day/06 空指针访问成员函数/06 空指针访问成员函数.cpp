#include <iostream>

using namespace std;


class Person {
public:
	void showClassName() {
		cout << "class Name is Person" << endl;
	}

	void showAge() {
		// NULL -> m_Age;
		if (this == NULL) {
			return;
		}
		cout << "Age = " << this->m_Age << endl;
	}

	int m_Age;
};

void test01() {
	Person p1;
	p1.m_Age = 18;

	p1.showAge();
	p1.showClassName();

	// 以上正常，如果使用空指针
	Person *p2 = NULL;
	p2->showAge();	// 失败，因为实例化为NULL的空指针，但是函数存在this指针，导致调用失败
						// NULL->m_Age
						// 因此需要一些条件判断
	p2->showClassName(); // 成功，无参数

}

int main() {

	test01();
	system("pause");
	return EXIT_SUCCESS;
}