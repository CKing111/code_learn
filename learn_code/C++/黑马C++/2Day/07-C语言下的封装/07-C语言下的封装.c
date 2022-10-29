#include<stdio.h>
#include<string.h>
#include<stdlib.h>


// C语言封装中，属性和行为分开处理了，类型检测不够
// test02中，声明的Person对象，但是在C语言中可以传入Dog对象的函数


struct Person{
	char mName[64];
	int mAge;
};

void PersonEat(struct Person* p) {
	printf("%s在吃饭！\n", p->mName);
}

void test01() {
	struct Person p1;
	strcpy(p1.mName, "德玛西亚");

	PersonEat(&p1);
}


struct Dog {
	char mName[64];
	int mAge;
};
void DogEat(struct Dog * d) {
	printf("%s在吃狗粮！\n", d->mName);
}
void test02() {
	struct Dog d1;
	strcpy(d1.mName, "旺财");
	DogEat(&d1);

	// 声明的Person对象，但是在C语言中可以传入Dog对象的函数
	struct Person p1;
	strcpy(p1.mName, "老王");
	DogEat(&p1);
}
int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}