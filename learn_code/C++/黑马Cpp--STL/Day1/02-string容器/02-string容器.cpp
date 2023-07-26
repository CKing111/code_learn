#include<iostream>
#include<string>
#include<stdexcept>
#include<vector>
using namespace std;


void test01() {
	/*
		string();//����һ���յ��ַ��� ����: string str;      
		string(const string& str);//ʹ��һ��string�����ʼ����һ��string����
		string(const char* s);//ʹ���ַ���s��ʼ��
		string(int n, char c);//ʹ��n���ַ�c��ʼ�� 

	*/ 
	string s1;// �������ַ���
	string s2(s1);// �������죬ʹ��string��ʼ��
	string s3("aaa");//�вι���	����Ӧchar��ʼ��
	string s4(10, 'c');	// �����������вι��죬��ʼ��10��c�ַ�

	cout << s3 << endl;
	cout << s4 << endl;

	/*
		string& operator=(const char* s);//char*�����ַ��� ��ֵ����ǰ���ַ���
		string& operator=(const string &s);//���ַ���s������ǰ���ַ���
		string& operator=(char c);//�ַ���ֵ����ǰ���ַ���
		string& assign(const char *s);//���ַ���s������ǰ���ַ���
		string& assign(const char *s, int n);//���ַ���s��ǰn���ַ�������ǰ���ַ���
		string& assign(const string &s);//���ַ���s������ǰ�ַ���
		string& assign(int n, char c);//��n���ַ�c������ǰ�ַ���
		string& assign(const string &s, int start, int n);//��s��start��ʼn���ַ���ֵ���ַ���
	*/
	// ��ֵ����
	string s5;
	s5 = s4;  // string��ֵ
	// string& assign(const char *s, int n);//���ַ���s��ǰn���ַ�������ǰ���ַ���
	s5.assign("abcdefg", 3);  // ��ȡ��ֵ
	cout << "s5.assign('abcdefg', 3) = " << s5 << endl;

	// //��s��start��ʼn���ַ���ֵ���ַ���
	string s6 = "abcedffg";
	string s7;

	s7.assign(s6, 5, 3);
	cout << "s7.assign(s6, 5, 3) = " << s7 << endl;
}

/*
��ȡ�ַ�����
char& operator[](int n);//ͨ��[]��ʽȡ�ַ�
char& at(int n);//ͨ��at������ȡ�ַ�

���� at()����Խ��ʱ���׳�һ���쳣   out_of_range
		[] ����Խ��ֱ�ӹҵ�
*/
void test02() {
	string s = "hello world!!";

	for (int i = 0; i < s.size(); i++) {
		cout << s[i] << endl;
	}
	cout << ".at()ѭ����" << endl;
	for (int i = 0; i < s.size(); i++) {
		cout << s.at(i) << endl;
	}

	cout << "����Խ�磺" << endl;
	try {
		s.at(100);
	}
	catch (exception &e) {			 // out_of_range
		cout << e.what() << endl;
	}
}



/*
	ƴ�Ӳ���
	string& operator+=(const string& str);//����+=������
	string& operator+=(const char* str);//����+=������
	string& operator+=(const char c);//����+=������
	string& append(const char *s);//���ַ���s���ӵ���ǰ�ַ�����β
	string& append(const char *s, int n);//���ַ���s��ǰn���ַ����ӵ���ǰ�ַ�����β
	string& append(const string &s);//ͬoperator+=()
	string& append(const string &s, int pos, int n);//���ַ���s�д�pos��ʼ��n���ַ����ӵ���ǰ�ַ�����β
	string& append(int n, char c);//�ڵ�ǰ�ַ�����β���n���ַ�c
*/ 

void test03() {
	// 	string& operator+=(const string& str);//����+=����������������
	string str1 = "��";
	string str2 = "��";
	string str3 = "��";

	str1 += str2;
	cout << str1 << endl;
	str1 += str3;
	cout << str1 << endl;

	// string& append(const string &s);//ͬoperator+=()
	str1.append(str3);
	cout << str1 << endl;
 }


/*
		�ַ������Һ��滻
		int find(const string& str, int pos = 0) const; //����str��һ�γ���λ��,��pos��ʼ����
		int find(const char* s, int pos = 0) const;  //����s��һ�γ���λ��,��pos��ʼ����
		int find(const char* s, int pos, int n) const;  //��posλ�ò���s��ǰn���ַ���һ��λ��
		int find(const char c, int pos = 0) const;  //�����ַ�c��һ�γ���λ��
		int rfind(const string& str, int pos = npos) const;//����str���һ��λ��,��pos��ʼ����
		int rfind(const char* s, int pos = npos) const;//����s���һ�γ���λ��,��pos��ʼ����
		int rfind(const char* s, int pos, int n) const;//��pos����s��ǰn���ַ����һ��λ��
		int rfind(const char c, int pos = 0) const; //�����ַ�c���һ�γ���λ��
		string& replace(int pos, int n, const string& str); //�滻��pos��ʼn���ַ�Ϊ�ַ���str
		string& replace(int pos, int n, const char* s); //�滻��pos��ʼ��n���ַ�Ϊ�ַ���s

*/
void test04() {
	// ����
	string str = "abc�Ұ��������찲��defglmh";
	int pos1 = str.find("bd", 0);	  // ����Ҳ����ַ����ͻ᷵��-1���ҵ��ͷ��ص�һ�γ��ֵ�λ��
	cout << pos1 << endl;

	int pos2 = str.find("��", 0);	 // �ӵ�0δ֪��ʼ���ң���һ�γ��ֵ�δ֪
	cout << pos2 << endl;

	int pos3 = str.rfind("��", -1);	 // �����һ��λ�ÿ�ʼ���ң����ŵ�һ�γ��ֵ�λ��
	cout << pos3 << endl << endl;

	// �滻
	// string& replace(int pos, int n, const string& str); //�滻��pos��ʼn���ַ�Ϊ�ַ���str
	string str2 = "����";
	str.replace(str.find("��", 0), 2, str2);
	cout << str.find("��", 0) << endl << str << endl;	// abc�Ҳ����������찲��defglmh
}



/*
	compare������>ʱ���� 1��<ʱ���� -1��==ʱ���� 0��
	�Ƚ����ִ�Сд���Ƚ�ʱ�ο��ֵ�˳����Խǰ���ԽС��
	��д��A��Сд��aС��

	int compare(const string &s) const;//���ַ���s�Ƚ�
	int compare(const char *s) const;//���ַ���s�Ƚ�
*/
void test05() {
	string str1 = "abcde";
	string str2 = "abcde";
	if (str1.compare(str2) == 0) {
		cout << "str1 = str2;" << endl;
	}
	else if (str1.compare(str2) > 0) {
		cout << "str1 > str2;" << endl;
	}
	else {
		cout << "str1 < str2;" << endl;
	}
}

/*
	����string���Ӵ�
	string substr(int pos = 0, int n = npos) const;//������pos��ʼ��n���ַ���ɵ��ַ���
*/ 
void test06() {
	string str1 = "dsfdghj�Ұ�����kljertyuvxc";
	cout << str1.substr(str1.find("�Ұ�����"), 8) << endl;// һ�����������ֽ�

	string email = "cuixiaokai@outlook.com";
	int pos = email.find("@");
	string userName = email.substr(0, pos);
	cout << "userName : " << userName << endl << endl;


	// Ӧ�ã�����ַ�е�ÿ�����ʽ�ȡ��vector������
	// www itcast com cn
	string net = "www.itcast.com.cn";
	vector<string> v;
	//string��ȡ ,ѭ�� 
	int start=0;
	while (true) {
		// find���ص�pos��Ŀ�굽�������λ����
		int pos = net.find(".",start);
		// �жϽ���
		if (pos == -1) {
			// �����һ�����ʽ�ȡ ��ȡû�С�.����β��cn
			string tmp = net.substr(start, net.size() - start);
			v.push_back(tmp);
			break;
		}
		string tmp = net.substr(start, pos - start);

		v.push_back(tmp);
		start = pos + 1;
	}

	// ��ȡ
	for (vector<string>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << endl;//�����ã����string
	}
}

/*
	�����ɾ��
	string& insert(int pos, const char* s); //�����ַ���
	string& insert(int pos, const string& str); //�����ַ���
	string& insert(int pos, int n, char c);//��ָ��λ�ò���n���ַ�c
	string& erase(int pos, int n = npos);//ɾ����Pos��ʼ��n���ַ�
*/
void test07() {
	string str = "hello";
	str.insert(1, "111"); 
	cout << str << endl;  // h111ello

	// ɾ��111������erase
	str.erase(1, 3);
	cout << str << endl;
}

/* 
	string��char��ת��
	charתstring��ͨ��string���вι��켴��ʵ��
	stringתchar��c_str()��������string���͵ı���ת��Ϊconst char*���͵�ָ��
			ʹ��c_str()�������ص�ָ��ָ�����string�����ڲ������ݣ�������޸ĸ�ָ����ָ�������ʱ��Ӱ�쵽ԭʼ��string����
*/ 
void doWork(string s) {} 
void doWork(const char* s) {}
void test08() {
	// char* -> string
	char * str = "hello";
	string s(str);		// �вι���

	//string -> char*
	const char*str2 = s.c_str();

	doWork(str2);	 // ��������Ϊstring��������char*����������������ʽ����ת��
	  
	//doWork2(s);    // ����������������string��ʽת��Ϊ const char*
}																	  

/*
		С��ϰ��
				дһ�������������ڲ���string���ַ����е�����Сд��ĸ�ĳɴ�д
*/
void test09() {
	string str = "abCDeFg";
	for (int i = 0; i < str.size(); i++) {
		// Сдת��д
		//str[i] = toupper(str[i]);
		// ��дתСд
		str[i] = tolower(str[i]);
	}
	cout << str << endl;
}

int main() {

	//test01();
	//test02();
	//test03();
	//test04();
	//test05();
	//test06();
	//test07();
	test09();
	system("pause");
	return EXIT_SUCCESS;
}