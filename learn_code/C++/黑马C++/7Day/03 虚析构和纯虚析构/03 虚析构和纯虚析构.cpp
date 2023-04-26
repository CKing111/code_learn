#define _CRT_SECURE_NO_WARNINGS // ��Ӻ궨�壬���ñ���������

#include<iostream>
#include<string>

#include<cstring>
using namespace std;


// �����ڶ�������ʱ���������������ุ����������ʱ�������������������޷��������ã������ڴ�й©
// ����취����������Ϊ�����������virtual�ؼ���

// ��������
// ��ͬ�봿�麯��������������Ҫ������ҲҪ��ʵ�֣�����ʵ��
// ԭ����base������Ҳ�п����ж������ݣ�Ҳ��Ҫ���������ͷ��ڴ棬��������Ǵ��麯��ҲҪ��ʵ��
// ֻ�д�����������Ҳʱ�����࣬�޷�ʵ��������
class Animal {
public:
	Animal() {
		cout << "����Animal���캯��" << endl;
	};

	// �������������ж�������ʱ�����ܻ��ͷŲ��ɾ��������ڴ�й©
	//virtual ~Animal() {
	//	cout << "����Animal����������" << endl;
	//}

	// ��������
	virtual ~Animal() = 0;

	// �麯��
	virtual void speak() {
		cout << "������˵����" << endl;
	}
};

// ��������ʵ��
Animal::~Animal() {
	cout << "Animal�Ĵ�����������" << endl;
}

class Cat :public Animal {
public:
	// ���캯��
	Cat(const char* name) {
		cout << "����Cat���캯��" << endl;

		this->m_Name = new char[strlen(name) + 1];
		strcpy(this->m_Name, name);
	}

	virtual ~Cat() override {
		cout << "����Cat����������" << endl;
		if (this->m_Name != NULL) {
			delete [] this->m_Name;
			this->m_Name = NULL;
		}
	}

	virtual void speak() override {
		cout << "Сè��˵����" << endl;
	}

	char* m_Name; // Cat name
};

void test01() {
	Animal* animal = new Cat("Tom");
	animal->speak();

	delete animal;
}



int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}