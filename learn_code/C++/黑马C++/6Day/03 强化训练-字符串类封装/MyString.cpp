#include"MyString.h"

MyString::MyString(const char* str)
{
	// �вι���
	this->pString = new char[strlen(str) + 1];
	strcpy(this->pString, str);

	this->m_Size = strlen(str);
}

MyString::MyString(const MyString& str)
{
	// ��������
	this->pString = new char[strlen(str.pString) + 1];
	strcpy(this->pString, str.pString);

	this->m_Size = strlen(str.pString);
}

char& MyString::operator[](int index) {
	return this->pString[index];
}

// ���ظ�ֵ�����=
// 1.ֱ�Ӹ�ֵstr
MyString& MyString::operator=(char * str)
{
	// ���ж��Ƿ�Ϊ��
	if (this->pString != NULL) {
		delete this->pString;
		this->pString = NULL;
	}
	// ���
	this->pString = new char[strlen(str) + 1];
	strcpy(this->pString, str);

	this->m_Size = strlen(str);
	return *this;
}
// 2.ֱ�Ӹ�ֵ�����
MyString& MyString::operator=(MyString& str)
{
	// ���ж��Ƿ�Ϊ��
	if (this->pString != NULL) {
		delete this->pString;
		this->pString = NULL;
	}
	// ���
	this->pString = new char[strlen(str.pString) + 1];
	strcpy(this->pString, str.pString);

	this->m_Size = strlen(str.pString);
	return *this;
}

// ���ؼӺ�
// 1.
MyString MyString::operator+(char* str) {
	// �����ռ�
	int newSize = this->m_Size + strlen(str) + 1;
	char* temp = new char[newSize];

	// temp�ռ丳ֵ
	memset(temp, 0, newSize);		// ��տռ�
	strcat(temp, this->pString);
	strcat(temp, str);

	// ��ʱ�ռ����¹��������
	MyString newString(temp);		// �вι���
	delete[] temp;					// �ͷ���ʱ�ռ�

	return newString;
}
// 2.
MyString MyString::operator+(MyString& str) {
	// �����ռ�
	int newSize = this->m_Size + strlen(str.pString) + 1;
	char* temp = new char[newSize];

	// temp�ռ丳ֵ
	memset(temp, 0, newSize);		// ��տռ�
	strcat(temp, this->pString);
	strcat(temp, str.pString);

	// ��ʱ�ռ����¹��������
	MyString newString(temp);		// �вι���
	delete[] temp;					// �ͷ���ʱ�ռ�

	return newString;
}

// ����==
// 1.
bool MyString::operator==(char* str) {
	if (strcmp(this->pString, str) == 0) {
		return true;
	}
	return false;
}
// 2. 
bool MyString::operator==(MyString& str) {
	if (strcmp(this->pString, str.pString) == 0) {
		return true;
	}
	return false;
}






MyString::~MyString()
{
	// ��������
	if (this->pString != NULL) {
		delete [] this->pString;
		this->pString = NULL;
	}
}



// ȫ�ֺ�������<<
ostream& operator<<(ostream& cout, MyString str) {
	cout << str.pString;		// ע�����س�Ա����ʹ����Ԫ
	return cout;
}

// ȫ�ֺ���������>>
istream& operator>>(istream& cin, MyString& str) {
	// �ж�ԭʼ�����Ƿ������ݣ���������ͷŵ�
	if (str.pString != NULL) {
		delete str.pString;
		str.pString = NULL;
	}

	// �������û�����
	char buf[1024];			// ������
	cin >> buf;				// �û�����ת�뻺��

	// ��buf���ݷ����ڲ�ά��pString��
	str.pString = new char[strlen(buf) + 1];
	strcpy(str.pString, buf);

	str.m_Size = strlen(buf);

	return cin;
}