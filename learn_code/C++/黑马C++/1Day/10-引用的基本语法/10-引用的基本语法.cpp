#include<iostream>

using namespace std;

// ���þ�������������洢�ռ���һ�����Ե��õı���
// ���ÿ��Կ����Ǽ������Ƶ�ָ�룬��ָ�밲ȫ
// ��&���ţ��ڱ������������ã��Ҳ����ȡ��ַ�������ã�
// 1.�����﷨��Type &���� = ԭ��
void test01() {
	int a = 10;
	int& b = a; // �β�b = ʵ��a,����b�ȼ��ڲ���a

	b = 20; //�ı����ã���ԭʼ��ֵַҲ�����ı䣬aҲ�ı�

	cout << "a = " << a << endl;
	cout << "&b = " << b << endl;
}
// 2.���ñ����ʼ��
void test02() {
	//int& a;   �����ԣ�δ��ʼ��
	int a = 10;
	int& b = a; //���ó�ʼ����Ͳ����޸���,���ܱ�ɱ��˵ı���

	int c = 20;
	//b = c;   //���Ǹ�ֵ���ǿ��Ըı�b��
	//int& b = c;  // �����ԣ���b��: �ض��壻��γ�ʼ��
	cout << "b = " << b << endl;
}
// 3.��������н�������
void test03() {
	// ��ʼ�����顣
	int arr[10];
	for (int i = 0; i < 10; i++) {
		arr[i] = i;
	}

	// ���������ñ���
	// ��һ�ַ�ʽ
	int(&pArr)[10] = arr;
	// ��ӡ
	for (int i = 0; i < 10; i++) {
		cout << pArr[i] << " " << endl;
	}

	// �ڶ��ַ�ʽ
	typedef int(ARRAYREF)[10];		//��ʾ������һ������10��Ԫ�ص�int��������
									// typedef ΪC���ԵĹؼ��֣�������Ϊһ���������Ͷ���һ��������
	ARRAYREF& pArr2 = arr;
	// ��ӡ
	for (int i = 0; i < 10; i++) {
		cout << pArr2[i] << " " << endl;
	}
}


int main() {
	//test01();
	//test02();
	test03();
	system("pause");
	return EXIT_SUCCESS;

}