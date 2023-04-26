#pragma once
#include <iostream>
using namespace std;

class MyArray {
public:
	MyArray();						// 默认构造
	MyArray(int capacity);			// 有参构造
	MyArray(const MyArray& arr);	// 拷贝构造
	~MyArray();						// 析构函数

	void pushBack(int val);			// 尾部插入数值
	void setData(int index, int val);	// 根据位置设置数据
	int getData(int index);		// 根据位置获取数据
	int getSize();					// 获取数组大小
	int getCapacity();				// 获取数组容量
private:
	
	int* pAddress;					// 指向堆区的数据指针
	int m_Capacity;					// 数组容量
	int m_Size;						// 数组大小
};