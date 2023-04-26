#include "MyArray.h"

MyArray::MyArray()
{
	// 默认构造函数，初始化参数
	cout << "使用MyArray默认构造函数" << endl;
	this->m_Capacity = 100;
	this->m_Size = 0;
	this->pAddress = new int[this->m_Capacity];		// 开辟空间
}

MyArray::MyArray(int capacity)
{
	// 有参构造，根据输入容量大小
	cout << "使用MyArray有参构造函数" << endl;
	this->m_Capacity = capacity;
	this->m_Size = 0;
	this->pAddress = new int[this->m_Capacity];		// 开辟空间
}

MyArray::MyArray(const MyArray& arr)
{
	// 拷贝构造函数，同类型可直接访问
	cout << "使用MyArray拷贝构造函数" << endl;
	this->m_Size = arr.m_Size;
	this->m_Capacity = arr.m_Capacity;

	this->pAddress = new int[this->m_Capacity];	//开辟空间
	// copy数组数据
	for (int i = 0; i < m_Size; i++) {
		this->pAddress[i] = arr.pAddress[i];
	}
}

MyArray::~MyArray()
{
	// 析构函数
	cout << "使用析构函数" << endl;
	if(this->pAddress != NULL){
		delete[] this->pAddress;		// 释放new空间，【】表示这里是多元素数组
		this->pAddress = NULL;			// 
	}
}

void MyArray::pushBack(int val)
{
	// 尾插
	this->pAddress[this->m_Size] = val;		// 尾元素索引
	this->m_Size++;							// 更新数组大小
}

void MyArray::setData(int index, int val)
{
	// 设置数据
	this->pAddress[index] = val;
}

int MyArray::getData(int index)
{
	// 获取数据
	return this->pAddress[index];
}

int MyArray::getSize()
{
	// 获取数组size
	return this->m_Size;
}

int MyArray::getCapacity()
{
	// 获取容量
	return this->m_Capacity;
}
