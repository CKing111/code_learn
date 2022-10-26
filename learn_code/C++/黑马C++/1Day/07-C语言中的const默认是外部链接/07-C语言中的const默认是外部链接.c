#include<stdio.h>
#include<string.h>
#include<stdlib.h>

int main() {
	extern const int a;		// 自动去搜索外部变量
	printf("a = %d", a);
	system("pause");
	return EXIT_SUCCESS;
}