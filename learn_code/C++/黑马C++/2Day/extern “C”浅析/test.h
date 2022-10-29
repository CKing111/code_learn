#pragma once

// 当C++需要调用多个C函数时
#ifdef __cplusplus	// 两个下划线
		// __cplusplus是cpp中的自定义宏，
		// 那么定义了这个宏的话表示这是一段cpp的代码，也就是说，
		// 上面的代码的含义是:如果这是一段cpp的代码，那么加入extern "C"{和}处理其中的代码。
extern "C"{
#endif

#include<stdio.h>

void show();
void show2();
void show3();

#ifdef __cplusplus
}
#endif