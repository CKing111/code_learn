#include<iostream>

using namespace std;

// C++ �쳣����

int myDivide(int a, int b) {
	// C ���Դ�����
	if (b == 0) {
		//return -1;

		// C++ �� �쳣���׳�
		throw 1;
	}
	return a / b;
}

void test01() {
	int a = 10;
	int b = 0;
	// C���Դ���ʽ
	myDivide(a, b);

	// C++ �����쳣
	try {		// ����ĳ�����쳣�������ú��������׳��쳣
		int ret = myDivide(a, b);
		cout << "ret �Ľ��Ϊ��" << ret << endl;
	}
	catch (int ) {		// �������׳����쳣
		cout << "int ���͵��쳣������" << endl;
	}
}



int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}