#include<iostream>
#include<fstream>
using namespace std;

// д�ļ�
void test01() {
	// ���ļ�
	// ���� �� ���ļ�·�� �� �򿪷�ʽ��
	// ����1��
	ofstream ofs;		// д���ļ��Ĳ�����ڣ�ofstream
	ofs.open("./1.txt", ios::out | ios::trunc);
	// ����2��
	//fstream ofs("./1.txt", ios::out | ios::trunc);

	// �ж��ļ��Ƿ�򿪳ɹ�
	if (!ofs.is_open()) {
		cout << "�ļ���ʧ��" << endl;
		return;
	}

	// д�ļ�
	ofs << "������XXX" << endl;
	ofs << "���䣺XX" << endl;

	// �ر������󣬹ر��ļ�
	ofs.close();

}

// ���ļ�
void test02() {
	ifstream ifs;
	ifs.open("./1.txt", ios::in);

	if (!ifs) {
		cout << "���ļ�ʧ�ܣ�" << endl;
		return;
	}

	//// ����1��
	//char buf[1024] = { 0 };
	//// ��ÿ��������뵽������
	//while (ifs >> buf) {		// ���ж�ȡ��ֱ���ļ�β��
	//	cout << buf << endl;
	//}

	//// ����2��
	//char buff[1024] = { 0 };
	//while (!ifs.eof()) {		// .eof()��ʾ�ж��Ƿ�Ϊ�ļ�β��
	//	ifs.getline(buff, sizeof(buff));
	//	cout << buff << endl;
	//}

	// ����3�������ַ���ȡ
	char c;
	while ((c = ifs.get()) != EOF) {
		cout << c;
	}


	// �ر�������
	ifs.close();
}

int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}