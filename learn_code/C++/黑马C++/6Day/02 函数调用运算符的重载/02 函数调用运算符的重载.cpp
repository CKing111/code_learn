#include <iostream>;

using namespace std;


class MyFunc {
public:

	// ����С���ţ�����������strʱ��ֱ�����
	void operator()(string text) {
		cout << text << endl;
	}
};

void test01() {

	// �º��������ʻ������أ�������ģ�º������
	MyFunc func;
	func("hello world!!");

}

class MyAdd {
public:
	int operator()(int a, int b) {
		return a + b;
	}
};

void test02(){
	MyAdd add;
	cout << add(1, 2) << endl;

	// ����������
	// ֱ��ʹ�����أ���
	cout << MyAdd()(10, 10) << endl;
}

int main() {
	test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}