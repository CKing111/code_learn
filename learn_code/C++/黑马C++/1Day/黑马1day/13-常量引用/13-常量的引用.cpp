#include<iostream>

using namespace std;

// 1.����������
// �����ã���������
void test01() {
	// int &ref = 10; // �����ԣ������˲��Ϸ����ڴ�
	const int& ref = 10; // ���ԣ�����const����������Զ�������ʱ�ڴ棬��Ϊ�Ϸ��ڴ�
		// �ȼ��ڣ� int tmp = 10; const int &ref = tmp;
	cout << "ԭʼref = " << ref << endl;

	// ֻҪ�ǺϷ��ռ��ڴ�Ķ����Խ��и�ֵ���޸���ֵ
	// ref = 20;  // �����ԣ�const�̶��˳���
	// ʹ��ָ���ƿ��༭�������޸�
	int* p = (int*)&ref;
	*p = 1000;
	cout << "ref = " << ref << endl;
}

// 2.�����βΣ���������ʹ�ó�����
// ֻ��ʹ�ô�����βζ������޸�ʵ�Σ���const����
// ��ʾ���ɸ���
void showValue(const int &val) {
	// val += 1000;  // ���ɸ���
	cout << "value: " << val << endl;

	// �ɸģ����ǲ�����
	int* p = (int*)&val;
	*p = 1000;
	cout << "ָ���޸ĺ��value: " << val << endl;
}
void test02() {
	int a = 10;
	showValue(a);
}
int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}