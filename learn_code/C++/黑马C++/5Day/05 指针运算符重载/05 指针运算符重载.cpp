#include<iostream>

using namespace std;

class Person {
public:
	Person(int age) {
		this->m_Age = age;
		cout << "Person�вι��캯��" << endl;
	}

	void showAge() {
		cout << "����Ϊ��" << this->m_Age << endl;
	}

	~Person() {
		cout << "Person����������" << endl;
	}

private:
	int m_Age;
};

// ����ָ�� ���ࣩ
// �����й�new������ָ����ͷ�
class SmartPointer {
public:
	SmartPointer(Person* person) {			// �вι���
		this->person = person;
		cout << "SmartPointer���вι���" << endl;
	}

	// ����ָ���������ʹ���������ָ��һ������
	// 1.����->
	Person* operator->() {			// this = Person* person
		cout << "->����" << endl;
		return this->person;
	}
	// ���أ�*��
	Person & operator*() {		// �������ã������ر��壬��Ҫ��copy����
		cout << "(*)����" << endl;
		return *this->person;	// 
	}

	~SmartPointer() {						// �����������Զ��ͷ�
		cout << "SmartPointer�����������ͷ�ָ��" << endl;
		if (this->person != NULL) {
			delete this->person;
			this->person = NULL;
		}
	}
private:
	Person* person;
};
void test01() {

	// 1.ָ�빹��
	//Person* p = new Person(18);			// ����ָ��
	//delete p;							// �ͷ�ָ��
	//p->showAge();		// ָ�����
	//(*p).showAge();	// ָ�������
	//  2.����ָ�빹����
	SmartPointer sp = SmartPointer(new Person(18));		// �вι���
	// ����ָ�����
	// ����->ָ�����
	sp->showAge();		// ����Person*����������Ҫsp->->showAge()������������ʡ����
	// ���ؽ����ã�*��
	(*sp).showAge();
}
int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}