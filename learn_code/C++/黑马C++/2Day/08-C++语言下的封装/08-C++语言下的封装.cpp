#include<iostream>

#define _CRT_SECURE_NO_WARNINGS

using namespace std;


// C++�еķ�װ���ϸ�����ת����⣬���Ժ���Ϊ�Ƿ�װ��һ���
// 1.���Ժ���Ϊ��Ϊһ�����������д���
// 2.����Ȩ�ޣ�����Ȩ��public,����Ȩ��protected,˽��Ȩ��private
//		class��Ĭ����privateȨ��
//		struct��Ĭ��Ȩ����publicȨ��
// 3.Ȩ�ޣ�
//		˽��Ȩ�ޣ�����˽�г�Ա�����ԡ������������ڿ��Է��ʣ����ⲻ���Է��ʣ�����Ҳ�����Է��ʣ�
//		����Ȩ�ޣ��������ⶼ���Է��ʵģ�
//		����Ȩ�ޣ����ڲ����Է��ʣ���ǰ���������Է��ʣ����ⲿ�����Է���
struct Person {
	char mName[64];
	int mAge;
	void PersonEat() {
		cout << mName << "���˷���" << endl;
	}
};
struct Dog {
	char mName[64];
	int mAge;
	void DogEat() {
		cout << mName << "�Թ�����" << endl;
	}
};

void test01() {
	Person p1;
	strcpy_s(p1.mName, strlen("����") + 1, "����");
	p1.PersonEat();

	Dog d1;
	strcpy_s(d1.mName, strlen("����") + 1,"����");
	d1.DogEat();

	// p1.DogEat();		// ʧ�ܣ������ڵ�ǰʵ���ķ�װ����
}

class Animal {
// Ĭ��˽��Ȩ�ޣ��������ڷ���
	void eat() { mAge = 100; mHight = 180; mWeight = 70; };	//���ڷ���
	int mAge;
// ����Ȩ��
public:
	int mHight;
// ����Ȩ��
protected:
	int mWeight;
};

void test02() {
	Animal a1;
	//a1.eat();	// ˽��Ȩ��
	//a1.mAge = 100;	// ˽��Ȩ��
	a1.mHight = 180;	// ����Ȩ��
	//a1.mWeight = 70;	// ����Ȩ��
}
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;

}