#pragma once

// ��C++��Ҫ���ö��C����ʱ
#ifdef __cplusplus	// �����»���
		// __cplusplus��cpp�е��Զ���꣬
		// ��ô�����������Ļ���ʾ����һ��cpp�Ĵ��룬Ҳ����˵��
		// ����Ĵ���ĺ�����:�������һ��cpp�Ĵ��룬��ô����extern "C"{��}�������еĴ��롣
extern "C"{
#endif

#include<stdio.h>

void show();
void show2();
void show3();

#ifdef __cplusplus
}
#endif