#include<iostream>

using namespace std;

// ���õı�������C++�ڲ�ʵ�ֵ�ָ�볣��
// Type & ref = val;-----> Type* const ref = &val;
// ����һ�����ã�Ĭ������»��Զ�����һ��ָ������ֵ��ָ�볣��

void test01(int& ref){	// ���ô���
	ref = 100;			// ref�����ã�Ĭ��ת��Ϊ*ref = 100��ָ�븳ֵ
}

int main() {
	int a = 10;	
	int& aRef = a;		// �Զ�ת��Ϊint* const aRef = &a;��Ҳ˵�����ñ���Ҫ�г�ʼ��			
	aRef = 20;			// �Զ�ת��Ϊ*aRef = 20��ָ�븳ֵ
	cout << "a: " << a << endl;
	cout << "aRef: " << aRef << endl;
	test01(a);
	cout << "test01(a): " << a << endl;
	system("pause");
	return EXIT_SUCCESS;
}