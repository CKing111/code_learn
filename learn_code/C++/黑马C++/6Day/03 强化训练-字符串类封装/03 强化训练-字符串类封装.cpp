#include<iostream>
#include"MyString.h"

void test01() {
	MyString str("aaa");
	cout << str << endl;		// ����<<
	cout << "����������str!!" << endl;

	cin >> str;					// ����>>
	cout << "���������strΪ��"<<str << endl;

	MyString str2(str);
	cout << "���������str2Ϊ��" << str2 << endl;

	// �������ֵ
	cout << "str[1]Ϊ��" << str[1] << endl;		// ����[]

	str[1] = 'n';
	cout << "�޸�str[1]��Ϊ��" << str << endl;		// ����[]

	// ��ָ��ĵȺŸ�ֵ
	MyString str3 = " ";
	str3 = str;						// �Ⱥ�����
	cout << "���ظ�ֵ�������str3Ϊ��" << str3 << endl;		// ����[]

	// ���ؼӷ������
	MyString str4 = "abc";
	MyString str5 = "def";
	MyString str6 = str4 + str5;		// ����ֵ

	cout << "���ؼӷ��������str6Ϊ��" << str6 << endl;		// ����[]

	// ����==
	if (str == str3) {
		cout << "str �� str3��ȣ�" <<str<<", "<<str3 << endl;
	}
	else {
		cout << "str �� str3 ����ȣ�" << str << ", " << str3 << endl;
	}

}

int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}