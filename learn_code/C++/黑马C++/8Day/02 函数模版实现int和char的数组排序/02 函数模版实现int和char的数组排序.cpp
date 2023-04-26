#include<iostream>

using namespace std;

template<class T> 
void mySawp(T& a, T& b) {
	T temp = a;
	a = b;
	b = temp;
}
// Ŀ�꣺����ѡ������ʵ�ֶ�int��char���������
template<typename T>
void mySort(T arr[], int len) {
	for (int i = 0; i < len; i++) {
		int min = i;	// �Ե�ǰֵΪ��Сģ�棬ȥ�ȽϺ���ֵ��С
		for (int j = i + 1; j < len; j++) {
			if (arr[min] > arr[j]) {
				min = j;	// ��¼��Сֵ�±�
			}
		}
		// �ж���Сֵ�±��뿪ʼ�϶���i�Ƿ���ȣ������ͬ�ٽ���i��min���б�ֵ
		if (min != i) {
			mySawp(arr[i], arr[min]);		// ����
		}

	}
}

template<class T>
void printArray(T arr[], int len) {
	for (int i = 0; i < len; i++) {
		cout <<"��ǰ��"<<i+1<<"˳λֵΪ�� " << arr[i] << endl;
	}
}
void test01() {

	int arr[] = { 15, 2, 6, 23, 61 };
	
	int len = sizeof(arr) / sizeof(int);		// ����int�ĳ���
	mySort(arr, len);		// ����������ͳߴ�

	// ��ӡ����
	printArray(arr, len);


	// char�����������ӡ
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