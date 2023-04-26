#include<iostream>

using namespace std;

class MyInter {
	friend ostream& operator<<(ostream& cout, MyInter& myint);
public:
	MyInter() {
		this->m_Num = 0;
	}
	// ��Ա��������ǰ��++�����
	// ������������ã�������е������㣬Ŀ����һֱ�Ա�����м���
	MyInter& operator++() {
		// ��++
		m_Num++;
		// �󷵻�
		return *this;
	}
	// ��Ա�������غ���++�����
	MyInter operator++(int) {
		// �ȷ���
		MyInter temp = *this;	// �����ַ
		// ֵ��++
		++this->m_Num;
		// ���ؾ�ַ
		return temp;
	}
private:
	int m_Num;
};

ostream& operator<<(ostream& cout, MyInter& myint) {
	cout << myint.m_Num;
	return cout;
}


void test01() {
	MyInter myint;
	//cout << myint << endl;
	cout << "����ǰ��++��" << ++myint << endl;
	cout << myint << endl;
}

void test02() {
	MyInter myint2;
	//cout << "����ǰ��++��" << ++myint << endl;
	MyInter m2 = myint2++;
	cout << m2  << endl;
	m2++;
	cout << m2 << endl;
}
int main() {
	//test01();
	test02();
	system("pause");
	return EXIT_SUCCESS;
}