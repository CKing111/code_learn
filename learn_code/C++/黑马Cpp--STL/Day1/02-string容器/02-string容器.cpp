#include<iostream>
#include<string>
#include<stdexcept>
#include<vector>
using namespace std;


void test01() {
	/*
		string();//创建一个空的字符串 例如: string str;      
		string(const string& str);//使用一个string对象初始化另一个string对象
		string(const char* s);//使用字符串s初始化
		string(int n, char c);//使用n个字符c初始化 

	*/ 
	string s1;// 声明空字符串
	string s2(s1);// 拷贝构造，使用string初始化
	string s3("aaa");//有参构造	，适应char初始化
	string s4(10, 'c');	// 两个参数的有参构造，初始化10个c字符

	cout << s3 << endl;
	cout << s4 << endl;

	/*
		string& operator=(const char* s);//char*类型字符串 赋值给当前的字符串
		string& operator=(const string &s);//把字符串s赋给当前的字符串
		string& operator=(char c);//字符赋值给当前的字符串
		string& assign(const char *s);//把字符串s赋给当前的字符串
		string& assign(const char *s, int n);//把字符串s的前n个字符赋给当前的字符串
		string& assign(const string &s);//把字符串s赋给当前字符串
		string& assign(int n, char c);//用n个字符c赋给当前字符串
		string& assign(const string &s, int start, int n);//将s从start开始n个字符赋值给字符串
	*/
	// 赋值操作
	string s5;
	s5 = s4;  // string赋值
	// string& assign(const char *s, int n);//把字符串s的前n个字符赋给当前的字符串
	s5.assign("abcdefg", 3);  // 截取赋值
	cout << "s5.assign('abcdefg', 3) = " << s5 << endl;

	// //将s从start开始n个字符赋值给字符串
	string s6 = "abcedffg";
	string s7;

	s7.assign(s6, 5, 3);
	cout << "s7.assign(s6, 5, 3) = " << s7 << endl;
}

/*
存取字符操作
char& operator[](int n);//通过[]方式取字符
char& at(int n);//通过at方法获取字符

区别： at()访问越界时会抛出一个异常   out_of_range
		[] 访问越界直接挂掉
*/
void test02() {
	string s = "hello world!!";

	for (int i = 0; i < s.size(); i++) {
		cout << s[i] << endl;
	}
	cout << ".at()循环：" << endl;
	for (int i = 0; i < s.size(); i++) {
		cout << s.at(i) << endl;
	}

	cout << "测试越界：" << endl;
	try {
		s.at(100);
	}
	catch (exception &e) {			 // out_of_range
		cout << e.what() << endl;
	}
}



/*
	拼接操作
	string& operator+=(const string& str);//重载+=操作符
	string& operator+=(const char* str);//重载+=操作符
	string& operator+=(const char c);//重载+=操作符
	string& append(const char *s);//把字符串s连接到当前字符串结尾
	string& append(const char *s, int n);//把字符串s的前n个字符连接到当前字符串结尾
	string& append(const string &s);//同operator+=()
	string& append(const string &s, int pos, int n);//把字符串s中从pos开始的n个字符连接到当前字符串结尾
	string& append(int n, char c);//在当前字符串结尾添加n个字符c
*/ 

void test03() {
	// 	string& operator+=(const string& str);//重载+=操作符，返回引用
	string str1 = "我";
	string str2 = "爱";
	string str3 = "你";

	str1 += str2;
	cout << str1 << endl;
	str1 += str3;
	cout << str1 << endl;

	// string& append(const string &s);//同operator+=()
	str1.append(str3);
	cout << str1 << endl;
 }


/*
		字符串查找和替换
		int find(const string& str, int pos = 0) const; //查找str第一次出现位置,从pos开始查找
		int find(const char* s, int pos = 0) const;  //查找s第一次出现位置,从pos开始查找
		int find(const char* s, int pos, int n) const;  //从pos位置查找s的前n个字符第一次位置
		int find(const char c, int pos = 0) const;  //查找字符c第一次出现位置
		int rfind(const string& str, int pos = npos) const;//查找str最后一次位置,从pos开始查找
		int rfind(const char* s, int pos = npos) const;//查找s最后一次出现位置,从pos开始查找
		int rfind(const char* s, int pos, int n) const;//从pos查找s的前n个字符最后一次位置
		int rfind(const char c, int pos = 0) const; //查找字符c最后一次出现位置
		string& replace(int pos, int n, const string& str); //替换从pos开始n个字符为字符串str
		string& replace(int pos, int n, const char* s); //替换从pos开始的n个字符为字符串s

*/
void test04() {
	// 查找
	string str = "abc我爱北京爱天安门defglmh";
	int pos1 = str.find("bd", 0);	  // 如果找不到字符串就会返回-1，找到就返回第一次出现的位置
	cout << pos1 << endl;

	int pos2 = str.find("爱", 0);	 // 从第0未知开始查找，第一次出现的未知
	cout << pos2 << endl;

	int pos3 = str.rfind("爱", -1);	 // 从最后一个位置开始查找，倒着第一次出现的位置
	cout << pos3 << endl << endl;

	// 替换
	// string& replace(int pos, int n, const string& str); //替换从pos开始n个字符为字符串str
	string str2 = "不爱";
	str.replace(str.find("爱", 0), 2, str2);
	cout << str.find("爱", 0) << endl << str << endl;	// abc我不爱北京爱天安门defglmh
}



/*
	compare函数在>时返回 1，<时返回 -1，==时返回 0。
	比较区分大小写，比较时参考字典顺序，排越前面的越小。
	大写的A比小写的a小。

	int compare(const string &s) const;//与字符串s比较
	int compare(const char *s) const;//与字符串s比较
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
	返回string的子串
	string substr(int pos = 0, int n = npos) const;//返回由pos开始的n个字符组成的字符串
*/ 
void test06() {
	string str1 = "dsfdghj我爱北京kljertyuvxc";
	cout << str1.substr(str1.find("我爱北京"), 8) << endl;// 一个中文两个字节

	string email = "cuixiaokai@outlook.com";
	int pos = email.find("@");
	string userName = email.substr(0, pos);
	cout << "userName : " << userName << endl << endl;


	// 应用：将网址中的每个单词截取到vector容器中
	// www itcast com cn
	string net = "www.itcast.com.cn";
	vector<string> v;
	//string截取 ,循环 
	int start=0;
	while (true) {
		// find返回的pos是目标到出发点的位置数
		int pos = net.find(".",start);
		// 判断结束
		if (pos == -1) {
			// 将最后一个单词截取 读取没有‘.’结尾的cn
			string tmp = net.substr(start, net.size() - start);
			v.push_back(tmp);
			break;
		}
		string tmp = net.substr(start, pos - start);

		v.push_back(tmp);
		start = pos + 1;
	}

	// 读取
	for (vector<string>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << endl;//解引用，输出string
	}
}

/*
	插入和删除
	string& insert(int pos, const char* s); //插入字符串
	string& insert(int pos, const string& str); //插入字符串
	string& insert(int pos, int n, char c);//在指定位置插入n个字符c
	string& erase(int pos, int n = npos);//删除从Pos开始的n个字符
*/
void test07() {
	string str = "hello";
	str.insert(1, "111"); 
	cout << str << endl;  // h111ello

	// 删除111，利用erase
	str.erase(1, 3);
	cout << str << endl;
}

/* 
	string和char的转换
	char转string：通过string的有参构造即可实现
	string转char：c_str()函数将该string类型的变量转换为const char*类型的指针
			使用c_str()函数返回的指针指向的是string对象内部的数据，因此在修改该指针所指向的内容时会影响到原始的string对象。
*/ 
void doWork(string s) {} 
void doWork(const char* s) {}
void test08() {
	// char* -> string
	char * str = "hello";
	string s(str);		// 有参构造

	//string -> char*
	const char*str2 = s.c_str();

	doWork(str2);	 // 函数输入为string，而给他char*，编译器进行了隐式类型转换
	  
	//doWork2(s);    // 报错，编译器不会想string隐式转换为 const char*
}																	  

/*
		小练习：
				写一个函数，函数内部将string的字符串中的所有小写字母改成大写
*/
void test09() {
	string str = "abCDeFg";
	for (int i = 0; i < str.size(); i++) {
		// 小写转大写
		//str[i] = toupper(str[i]);
		// 大写转小写
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