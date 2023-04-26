#include<iostream>

int main() {
	int a[5] = { 1,2,3,4,5 };
	int* prt = a;

	for (int a = 0; i < 5; i++) {
		std::cout << "a[" << i << "] = " << *prt << std::endl;
		prt++;
	}
	/*
	在以上代码中，我们定义了一个大小为5的整型数组a，
	并将指针ptr指向数组的首地址。使用for循环遍历数组时，
	每次输出当前指针所指向的元素的值，并将指针加1，指向下一个元素。
	*/
	return 
}