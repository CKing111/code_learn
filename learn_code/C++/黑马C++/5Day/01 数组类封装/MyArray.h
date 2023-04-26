#pragma once
#include <iostream>
using namespace std;

class MyArray {
public:
	MyArray();						// Ĭ�Ϲ���
	MyArray(int capacity);			// �вι���
	MyArray(const MyArray& arr);	// ��������
	~MyArray();						// ��������

	void pushBack(int val);			// β��������ֵ
	void setData(int index, int val);	// ����λ����������
	int getData(int index);		// ����λ�û�ȡ����
	int getSize();					// ��ȡ�����С
	int getCapacity();				// ��ȡ��������
private:
	
	int* pAddress;					// ָ�����������ָ��
	int m_Capacity;					// ��������
	int m_Size;						// �����С
};