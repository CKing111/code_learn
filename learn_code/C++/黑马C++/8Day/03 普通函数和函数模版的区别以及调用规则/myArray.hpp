#pragma once
#include<iostream>
using namespace std;

// �����������ģ��
template<class T>
class MyArray {
public:
	// �вι���
	explicit MyArray(int capacity) {		// explicit��������ֹ��ʽ����ת���Ĺؼ��֡�
		this->m_Capacity = capacity;
		this->m_Size = 0;
		this->pAddress = new T[this->m_Capacity];
	}

	// �������캯��
	MyArray(const MyArray& arr) {
		this->m_Capacity = arr.m_Capacity;
		this->m_Size = arr.m_Size;

		// ���������
		this->pAddress = new T[this->m_Capacity];
		for (int i = 0; i < m_Size; i++) {
			this->pAddress[i] = arr.pAddress[i];
		}
	}

	// ����=
	MyArray& operator=(const MyArray& arr) {
		// �ж��������ݣ���������ͷŵ�
		if (this->pAddress != NULL) {
			delete[] this->pAddress;
			this->pAddress = NULL;
		}

		this->m_Capacity = arr.m_Capacity;
		this->m_Size = arr.m_Size;

		// ���������
		this->pAddress = new T[this->m_Capacity];
		for (int i = 0; i < m_Size; i++) {
			this->pAddress[i] = arr.pAddress[i];
		}
		return *this;
	}


	// ���� []  ʵ�֣�MyArray arr (100); arr[0] = 0;
	T& operator[](int index) {		// T& ��ʾ����arr[0]�󣬻���Ϊ��ֵ���ڣ����Ա�ʹ��
		return this->pAddress[index];
	}

	// β�巨
	void pushBack(const T& val) {
		// �ж��Ƿ񳬳�����
		if (this->m_Capacity == this->m_Size) {
			return;
		}

		this->pAddress[this->m_Size] = val;		// ��β�����и�ֵ
		this->m_Size++;
	}

	// βɾ��
	void popBack() {
		if (this->m_Size == 0) {
			return;
		}

		this->m_Size--;		// ʹ�޷����ʣ�������ֱֵ�Ӹ���
	}

	// ���������С
	int getSize() {
		return this->m_Size;
	}

	// ��������
	~MyArray����{
		if (this->pAddress != NULL) {
			delete[] this->pAddress;
}			this->pAddress = NULL;
	}
private:
	T* pAddress;	// ��ʵ���ٵ��������ݵ�ָ��

	// ��������
	int m_Capacity;

	// ���鳤��
	int m_Size;
};
