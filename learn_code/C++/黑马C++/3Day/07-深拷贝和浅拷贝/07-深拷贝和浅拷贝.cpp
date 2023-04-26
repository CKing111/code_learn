#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<string>
using namespace std;

class Person {
public:
	// ���캯��
	Person() {}	//Ĭ�Ϲ���
	// �вι��죬Ŀ���ǳ�ʼ������
	Person(const char *name,int age) {
		cout << "�вι��캯��" << endl;
		m_Name = (char*)malloc(strlen(name) + 1);  // ���ٿռ�
					// strlen() ��һ�� C ��׼���е��ַ�����������
					// ���ڼ���һ���� null ��β���ַ����ĳ��ȣ������� null ��ֹ����
		strcpy(m_Name, name);	// <--
				// strcpy��C/C++��׼���е�һ���ַ�������������
				// ���ڽ�һ���ַ�������null�ַ���β�����Ƶ���һ���ַ����Ļ�������
				// ��������ָ��Ŀ���ָ��
		m_Age = age;
	}
	// ������ϵͳ���ṩĬ�ϼ�ֵ����

	// �������ͷ�������Լ����ϵ����ԣ�ָ�룬��̬����ռ䣩
	// ǳ�������������ͷ��ڴ�����ϵͳ������
	// ��Ϊǳ����ֻ������ַ�����ǻ��ͷŶ����ռ����Σ���Ҫʹ�����
	// �Լ��������������
	Person(const Person& p) {		// ���
		cout << "������캯��" << endl;
		m_Age = p.m_Age;
		m_Name = (char*)malloc(strlen(p.m_Name) + 1);	// �����ַ
		strcpy(m_Name, p.m_Name);	// cpyֵ
	}
	~Person() {
		cout << "�����������ã�" << endl;
		// �жϳ�Ա�����Ƿ���Ҫ�ͷţ�ָ���Ƿ�Ϊ��
		if (m_Name != NULL) {
			free(m_Name);	// malloc/free��C/C++���Եı�׼�⺯��,new/delete��malloc/free�������ʹ�á�
			m_Name = NULL;	// ʹָ��Ϊ�գ���ֹҰָ��
				/*
					���Ұָ�����ָ����ڴ��ַ��δ֪��(����ģ�����ȷ�ģ�û����ȷ���Ƶ�)��
					˵����ָ�����Ҳ�Ǳ������Ǳ����Ϳ������⸳ֵ�����ǣ�������ֵ��ֵ��ָ�����û�����壬
							��Ϊ������ָ��ͳ���Ұָ�룬��ָ��ָ���������δ֪
							(����ϵͳ�����������ָ��ָ����ڴ�����)��
					ע��Ұָ�벻��ֱ���������󣬲���Ұָ��ָ����ڴ�����Ż�����⡣
				*/
		}

	}

	// ����
	char* m_Name;		// ָ���������Ҫ���
	// ����
	int m_Age;
};

void test01() {
	Person p1("����", 10);		// �вι���
	Person p2(p1);				// �������

	cout << "����p2.name:" << p2.m_Name << ", p2.age:" << p2.m_Age << endl;

}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}