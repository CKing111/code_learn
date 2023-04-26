#include"MyString.h"

MyString::MyString(const char* str)
{
	// 有参构造
	this->pString = new char[strlen(str) + 1];
	strcpy(this->pString, str);

	this->m_Size = strlen(str);
}

MyString::MyString(const MyString& str)
{
	// 拷贝构造
	this->pString = new char[strlen(str.pString) + 1];
	strcpy(this->pString, str.pString);

	this->m_Size = strlen(str.pString);
}

char& MyString::operator[](int index) {
	return this->pString[index];
}

// 重载赋值运算符=
// 1.直接赋值str
MyString& MyString::operator=(char * str)
{
	// 先判断是否为空
	if (this->pString != NULL) {
		delete this->pString;
		this->pString = NULL;
	}
	// 深拷贝
	this->pString = new char[strlen(str) + 1];
	strcpy(this->pString, str);

	this->m_Size = strlen(str);
	return *this;
}
// 2.直接赋值类对象
MyString& MyString::operator=(MyString& str)
{
	// 先判断是否为空
	if (this->pString != NULL) {
		delete this->pString;
		this->pString = NULL;
	}
	// 深拷贝
	this->pString = new char[strlen(str.pString) + 1];
	strcpy(this->pString, str.pString);

	this->m_Size = strlen(str.pString);
	return *this;
}

// 重载加号
// 1.
MyString MyString::operator+(char* str) {
	// 声明空间
	int newSize = this->m_Size + strlen(str) + 1;
	char* temp = new char[newSize];

	// temp空间赋值
	memset(temp, 0, newSize);		// 清空空间
	strcat(temp, this->pString);
	strcat(temp, str);

	// 临时空间重新构造类对象
	MyString newString(temp);		// 有参构造
	delete[] temp;					// 释放临时空间

	return newString;
}
// 2.
MyString MyString::operator+(MyString& str) {
	// 声明空间
	int newSize = this->m_Size + strlen(str.pString) + 1;
	char* temp = new char[newSize];

	// temp空间赋值
	memset(temp, 0, newSize);		// 清空空间
	strcat(temp, this->pString);
	strcat(temp, str.pString);

	// 临时空间重新构造类对象
	MyString newString(temp);		// 有参构造
	delete[] temp;					// 释放临时空间

	return newString;
}

// 重载==
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
	// 析构函数
	if (this->pString != NULL) {
		delete [] this->pString;
		this->pString = NULL;
	}
}



// 全局函数重载<<
ostream& operator<<(ostream& cout, MyString str) {
	cout << str.pString;		// 注意隐藏成员参数使用友元
	return cout;
}

// 全局函数，重载>>
istream& operator>>(istream& cin, MyString& str) {
	// 判断原始输入是否有数据，如果有先释放掉
	if (str.pString != NULL) {
		delete str.pString;
		str.pString = NULL;
	}

	// 重新设置缓存区
	char buf[1024];			// 缓冲区
	cin >> buf;				// 用户输入转入缓存

	// 将buf数据放入内部维护pString中
	str.pString = new char[strlen(buf) + 1];
	strcpy(str.pString, buf);

	str.m_Size = strlen(buf);

	return cin;
}