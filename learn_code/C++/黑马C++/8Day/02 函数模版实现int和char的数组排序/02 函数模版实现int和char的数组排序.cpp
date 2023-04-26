#include<iostream>

using namespace std;

template<class T> 
void mySawp(T& a, T& b) {
	T temp = a;
	a = b;
	b = temp;
}
// 目标：利用选择排序实现对int和char数组的排序
template<typename T>
void mySort(T arr[], int len) {
	for (int i = 0; i < len; i++) {
		int min = i;	// 以当前值为最小模版，去比较后面值大小
		for (int j = i + 1; j < len; j++) {
			if (arr[min] > arr[j]) {
				min = j;	// 记录最小值下标
			}
		}
		// 判断最小值下标与开始认定的i是否相等，如果不同再交换i与min的列表值
		if (min != i) {
			mySawp(arr[i], arr[min]);		// 交换
		}

	}
}

template<class T>
void printArray(T arr[], int len) {
	for (int i = 0; i < len; i++) {
		cout <<"当前第"<<i+1<<"顺位值为： " << arr[i] << endl;
	}
}
void test01() {

	int arr[] = { 15, 2, 6, 23, 61 };
	
	int len = sizeof(arr) / sizeof(int);		// 数组int的长度
	mySort(arr, len);		// 排序传入数组和尺寸

	// 打印数组
	printArray(arr, len);


	// char类型排序与打印
	char charArr[] = "hellowworld";
	int charlen = sizeof(charArr) / sizeof(char);
	mySort(charArr, charlen);
	printArray(charArr, charlen);
}




int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}