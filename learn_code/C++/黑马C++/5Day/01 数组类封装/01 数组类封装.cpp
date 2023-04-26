#include<iostream>
#include "MyArray.h"
using namespace std;

void test01() {
	MyArray* arr = new MyArray(10);		// 有参构造开辟空间

	delete arr;

	MyArray arr2;						// 默认构造函数
	// 生成元素
	for (int i = 0; i < 10; i++) {
		arr2.pushBack(i + 100);
	}
	
	MyArray arr3(arr2);					// 拷贝构造函数

	// 设置元素
	arr3.setData(1, 1000);

	// 访问数组
	for (int i = 0; i < 10; i++) {
		cout << "arr3中，位置-" << i + 1 << "-的元素为：" << arr3.getData(i) << endl;
	}
	// 访问数组属性
	cout << "arr3数组的容量为：" << arr3.getCapacity() << endl;
	cout << "arr3数组的Size为：" << arr3.getSize() << endl;

}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}
