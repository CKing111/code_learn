#include<stdio.h>
#include<string.h>
#include<stdlib.h>


// C���Է�װ�У����Ժ���Ϊ�ֿ������ˣ����ͼ�ⲻ��
// test02�У�������Person���󣬵�����C�����п��Դ���Dog����ĺ���


struct Person{
	char mName[64];
	int mAge;
};

void PersonEat(struct Person* p) {
	printf("%s�ڳԷ���\n", p->mName);
}

void test01() {
	struct Person p1;
	strcpy(p1.mName, "��������");

	PersonEat(&p1);
}


struct Dog {
	char mName[64];
	int mAge;
};
void DogEat(struct Dog * d) {
	printf("%s�ڳԹ�����\n", d->mName);
}
void test02() {
	struct Dog d1;
	strcpy(d1.mName, "����");
	DogEat(&d1);

	// ������Person���󣬵�����C�����п��Դ���Dog����ĺ���
	struct Person p1;
	strcpy(p1.mName, "����");
	DogEat(&p1);
}
int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}