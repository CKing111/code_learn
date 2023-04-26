#pragma once
# define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<string>
using namespace std;

class MyString {
	// ��Ԫ������<<
	friend ostream& operator<<(ostream& cout, MyString str);
	// ��Ԫ������>>
	friend istream& operator>>(istream& cin, MyString& str);
public:
	//�вι���
	MyString(const char * str);
	/*�� C++ �У�const ��һ���ؼ��֣���ʾ�����������������α�����������������Ա�����ȵȡ������������ã�
	��ȫ�ԣ�ʹ�� const ���Է�ֹ����Ա�ڲ�������޸ı�����ֵ���Ӷ����ӳ���İ�ȫ�ԡ�
				���磬����㽫һ����������Ϊ const����ô�κγ����޸ĸó�������Ϊ���ᵼ�±������
	�Ż���ʹ�� const �����԰����������Գ�������Ż������һ������������Ϊ const����ô�������Ϳ��԰�ȫ�ؽ����Ż�Ϊһ��������
				�Ӷ���߳����Ч�ʡ�
	�ɶ��ԣ�ʹ�� const �������Ӵ���Ŀɶ��ԡ��ں���������ʹ�� const�����Ը��߶��߸ú��������޸ĸò�����
				�ڳ�Ա������ʹ�� const�����Ը��߶��߸ú��������޸Ķ����״̬��
	��֮��ʹ�� const ��һ�����õı��ϰ�ߣ�������߳���İ�ȫ�ԡ��ɶ��Ժ�Ч�ʡ�*/
	// ��������
	MyString(const MyString& str);

	// ����[]
	char& operator[](int index);

	// ����=
	MyString& operator=(char * str);			// str = "aaa"
	MyString& operator=(MyString& str);			// str2 = str

	// ����+��ʵ���ַ���ƴ�ӣ�����ֵ
	MyString operator+(char * str);
	MyString operator+(MyString& str);

	// ����==
	bool operator==(char* str);
	bool operator==(MyString& str);

	// ��������
	~MyString();
private:
	// ָ��������ַ���ָ��
	char* pString;

	// �ַ�������
	int m_Size;
};
