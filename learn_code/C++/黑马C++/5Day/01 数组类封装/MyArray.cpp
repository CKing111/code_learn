#include "MyArray.h"

MyArray::MyArray()
{
	// Ĭ�Ϲ��캯������ʼ������
	cout << "ʹ��MyArrayĬ�Ϲ��캯��" << endl;
	this->m_Capacity = 100;
	this->m_Size = 0;
	this->pAddress = new int[this->m_Capacity];		// ���ٿռ�
}

MyArray::MyArray(int capacity)
{
	// �вι��죬��������������С
	cout << "ʹ��MyArray�вι��캯��" << endl;
	this->m_Capacity = capacity;
	this->m_Size = 0;
	this->pAddress = new int[this->m_Capacity];		// ���ٿռ�
}

MyArray::MyArray(const MyArray& arr)
{
	// �������캯����ͬ���Ϳ�ֱ�ӷ���
	cout << "ʹ��MyArray�������캯��" << endl;
	this->m_Size = arr.m_Size;
	this->m_Capacity = arr.m_Capacity;

	this->pAddress = new int[this->m_Capacity];	//���ٿռ�
	// copy��������
	for (int i = 0; i < m_Size; i++) {
		this->pAddress[i] = arr.pAddress[i];
	}
}

MyArray::~MyArray()
{
	// ��������
	cout << "ʹ����������" << endl;
	if(this->pAddress != NULL){
		delete[] this->pAddress;		// �ͷ�new�ռ䣬������ʾ�����Ƕ�Ԫ������
		this->pAddress = NULL;			// 
	}
}

void MyArray::pushBack(int val)
{
	// β��
	this->pAddress[this->m_Size] = val;		// βԪ������
	this->m_Size++;							// ���������С
}

void MyArray::setData(int index, int val)
{
	// ��������
	this->pAddress[index] = val;
}

int MyArray::getData(int index)
{
	// ��ȡ����
	return this->pAddress[index];
}

int MyArray::getSize()
{
	// ��ȡ����size
	return this->m_Size;
}

int MyArray::getCapacity()
{
	// ��ȡ����
	return this->m_Capacity;
}
