#include<stdio.h>
#include<string.h>
#include<stdlib.h>

int main() {
	extern const int a;		// �Զ�ȥ�����ⲿ����
	printf("a = %d", a);
	system("pause");
	return EXIT_SUCCESS;
}