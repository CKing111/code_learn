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

	// �������������ʹ�ÿ�ָ��
	Person *p2 = NULL;
	p2->showAge();	// ʧ�ܣ���Ϊʵ����ΪNULL�Ŀ�ָ�룬���Ǻ�������thisָ�룬���µ���ʧ��
						// NULL->m_Age
						// �����ҪһЩ�����ж�
	p2->showClassName(); // �ɹ����޲���

}

int main() {

	test01();
	system("pause");
	return EXIT_SUCCESS;
}