#include<iostream>
#include "MyArray.h"
using namespace std;

void test01() {
	MyArray* arr = new MyArray(10);		// �вι��쿪�ٿռ�

	delete arr;

	MyArray arr2;						// Ĭ�Ϲ��캯��
	// ����Ԫ��
	for (int i = 0; i < 10; i++) {
		arr2.pushBack(i + 100);
	}
	
	MyArray arr3(arr2);					// �������캯��

	// ����Ԫ��
	arr3.setData(1, 1000);

	// ��������
	for (int i = 0; i < 10; i++) {
		cout << "arr3�У�λ��-" << i + 1 << "-��Ԫ��Ϊ��" << arr3.getData(i) << endl;
	}
	// ������������
	cout << "arr3���������Ϊ��" << arr3.getCapacity() << endl;
	cout << "arr3�����SizeΪ��" << arr3.getSize() << endl;

}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}
