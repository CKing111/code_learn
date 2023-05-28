#include<iostream>
#include<fstream>
using namespace std;

// 写文件
void test01() {
	// 打开文件
	// 参数 名 （文件路径 ， 打开方式）
	// 方法1：
	ofstream ofs;		// 写入文件的参数借口，ofstream
	ofs.open("./1.txt", ios::out | ios::trunc);
	// 方法2：
	//fstream ofs("./1.txt", ios::out | ios::trunc);

	// 判断文件是否打开成功
	if (!ofs.is_open()) {
		cout << "文件打开失败" << endl;
		return;
	}

	// 写文件
	ofs << "姓名：XXX" << endl;
	ofs << "年龄：XX" << endl;

	// 关闭流对象，关闭文件
	ofs.close();

}

// 读文件
void test02() {
	ifstream ifs;
	ifs.open("./1.txt", ios::in);

	if (!ifs) {
		cout << "打开文件失败！" << endl;
		return;
	}

	//// 方法1：
	//char buf[1024] = { 0 };
	//// 将每行输入读入到缓冲区
	//while (ifs >> buf) {		// 按行读取，直到文件尾部
	//	cout << buf << endl;
	//}

	//// 方法2：
	//char buff[1024] = { 0 };
	//while (!ifs.eof()) {		// .eof()表示判断是否为文件尾部
	//	ifs.getline(buff, sizeof(buff));
	//	cout << buff << endl;
	//}

	// 方法3：单个字符读取
	char c;
	while ((c = ifs.get()) != EOF) {
		cout << c;
	}


	// 关闭流对象
	ifs.close();
}

int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}