#include<iostream>

using namespace std;

int main() {
	extern const int a; //C++中，const无法自动搜索外部链接
	cout << a << endl;

	system("pause");
	return EXIT_SUCCESS;

}