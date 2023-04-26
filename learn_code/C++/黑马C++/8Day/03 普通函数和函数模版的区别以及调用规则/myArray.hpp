#pragma once
#include<iostream>
using namespace std;

// 声明数组的类模版
template<class T>
class MyArray {
public:
	// 有参构造
	explicit MyArray(int capacity) {		// explicit是用来防止隐式类型转换的关键字。
		this->m_Capacity = capacity;
		this->m_Size = 0;
		this->pAddress = new T[this->m_Capacity];
	}

	// 拷贝构造函数
	MyArray(const MyArray& arr) {
		this->m_Capacity = arr.m_Capacity;
		this->m_Size = arr.m_Size;

		// 堆区，深拷贝
		this->pAddress = new T[this->m_Capacity];
		for (int i = 0; i < m_Size; i++) {
			this->pAddress[i] = arr.pAddress[i];
		}
	}

	// 重载=
	MyArray& operator=(const MyArray& arr) {
		// 判断现有数据，如果有先释放掉
		if (this->pAddress != NULL) {
			delete[] this->pAddress;
			this->pAddress = NULL;
		}

		this->m_Capacity = arr.m_Capacity;
		this->m_Size = arr.m_Size;

		// 堆区，深拷贝
		this->pAddress = new T[this->m_Capacity];
		for (int i = 0; i < m_Size; i++) {
			this->pAddress[i] = arr.pAddress[i];
		}
		return *this;
	}


	// 重载 []  实现：MyArray arr (100); arr[0] = 0;
	T& operator[](int index) {		// T& 表示返回arr[0]后，还作为左值存在，可以被使用
		return this->pAddress[index];
	}

	// 尾插法
	void pushBack(const T& val) {
		// 判断是否超出容量
		if (this->m_Capacity == this->m_Size) {
			return;
		}

		this->pAddress[this->m_Size] = val;		// 最尾部序列赋值
		this->m_Size++;
	}

	// 尾删法
	void popBack() {
		if (this->m_Size == 0) {
			return;
		}

		this->m_Size--;		// 使无法访问，后续插值直接覆盖
	}

	// 返回数组大小
	int getSize() {
		return this->m_Size;
	}

	// 析构函数
	~MyArray（）{
		if (this->pAddress != NULL) {
			delete[] this->pAddress;
}			this->pAddress = NULL;
	}
private:
	T* pAddress;	// 真实开辟到堆区数据的指针

	// 数据容量
	int m_Capacity;

	// 数组长度
	int m_Size;
};
