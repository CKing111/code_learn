/*
	create by Cxk
	data : 2023.7.25
	createPerson(vector<Person>& v)创建五名选手，参数1：。。。。

*/

#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<deque>

#include<ctime>
using namespace std;

/*
		题目要求：
		有五名选手，选手ABCDE,10位评委分别对每一个选手打分，去除最高分，去除评委最低分，取平均分。
		1. 创建五名选手，放到vector容器中；
		2. 遍历vector容器，取出每一个选手，执行佛如循环，可以把10个评分打分存到deque容器中；
		3. sort算法对deque容器中分数排序，pop_back pop_front去除最高分和最低分；
		4. deque容器遍历，累加分数，累加分数/size()计算平均分；
		5. 让person.score = 平均分；
*/

// 选手类
class Person {
public:
	Person(string name, int score) {
		this->m_Name = name;
		this->m_Score = score;
	}
	string getName() { return m_Name; }
	int getScore() { return m_Score; }
	void setScore(int score) { m_Score = score; }
private:
	string m_Name;	// 姓名
	int m_Score;	// 平均分
};
void test01() {

}

void createPerson(vector<Person>&v) {
	// 选手名种子
	string nameSeed = "ABCDE";
	for (int i = 0; i < 5; i++) {
		string name = "选手";
		name += nameSeed[i];

		int score = 0;
		Person p(name, score);
		v.push_back(p);     // 测试1节点
	}

}
void printVector(vector<Person> v) {
	// 测试节点1
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++) {
		cout << "姓名：" << (*it).getName() << ", 成绩：" << (*it).getScore() << endl;
		// error:没有与这些操作数匹配的 "<<" 运算符	04 - 评委打分案例	d : \code\TrainingCode\黑马Cpp--STL\Day1\04 - 评委打分案例\04 - 评委打分案例.cpp	61
		// 没有声明string 头文件
	}
}
void printVectorAvg(vector<Person> v) {
	// 测试节点1
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++) {
		cout << "姓名：" << (*it).getName() << ", 平均成绩：" << (*it).getScore() << endl;
		// error:没有与这些操作数匹配的 "<<" 运算符	04 - 评委打分案例	d : \code\TrainingCode\黑马Cpp--STL\Day1\04 - 评委打分案例\04 - 评委打分案例.cpp	61
		// 没有声明string 头文件
	}
}
void printVector(deque<int> v) {
	// 测试节点1		
	for (deque<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it<<" ";
	}
	cout << endl;
}
void printVector(vector<int> v) {
	// 测试节点1		
	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

// 打分函数
void setScore(vector<Person>&v) {
	// 循环选手
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++) {
		// 有10个评委打分，创建打分容器
		deque<int>d;
		for (int i = 0; i < 10; i++) {
			int score = rand() % 41 + 60;    // 随机区间：60~99
			// %是取模运算符，表示将左边的数除以右边的数，然后取余数。因此，rand() % 40的结果是一个介于0和39之间的整数。
			// 因此，score = rand() % 40 + 60的结果就是将0到39之间的随机整数加上60, 得到一个60到100之间的随机整数。
			d.push_back(score);
		}
		//cout << "选手： " << it->getName() << "的打分结果：" << endl;
		//printVector(d);// 测试点2，重载
		//cout << endl;

		// 排序
		sort(d.begin(), d.end());		   // 默认从小到大
		//printVector(d);// 测试点2，重载	
        /*
				选手： 选手A的打分结果：
				60 77 80 74 82 81 99 62 85 88
				60 62 74 77 80 81 82 85 88 99
				选手： 选手B的打分结果：
				66 79 94 77 99 100 62 71 90 84
				62 66 71 77 79 84 90 94 99 100
				选手： 选手C的打分结果：
				61 68 67 90 65 60 97 80 98 70
				60 61 65 67 68 70 80 90 97 98
				选手： 选手D的打分结果：
				95 97 71 77 84 87 61 78 80 73
				61 71 73 77 78 80 84 87 95 97
				选手： 选手E的打分结果：
				63 91 99 94 62 91 72 63 64 81
				62 63 63 64 72 81 91 91 94 99
		 */

		// 去除最高分和最低分
		// 最高分
		d.pop_back();
		// 最低分
		d.pop_front();

		// 获取总分
		int sum = 0;
		for (deque<int>::iterator it = d.begin(); it != d.end(); it++) {
			sum += *it;
		}

		// 获取平均分
		int avg = sum / d.size();
		
		// 给对象赋值
		it->setScore(avg);
	}
}
int main() {

	// 设置随机数种子
	//srand((unsigned int)time(NULL));
	srand((unsigned int)10);
		// srand()函数来设置随机数生成器的种子值。
		// time(NULL)是一个系统函数，返回当前时间的时间戳(以秒为单位)。
		// 将这个时间戳转换为无符号整型后作为种子值传给srand()函数，可以保证每次运行程序时生成的随机数序列都不同。
	// 1. 声明存放容器
	vector<Person> v;
	// 2. 创建5名选手
	createPerson(v);
	//printVector(v);	// 测试1

	// 3. 打分
	setScore(v);

	// 4. 显示得分
	printVectorAvg(v);
	/*
		姓名：选手A, 平均成绩：74
		姓名：选手B, 平均成绩：74
		姓名：选手C, 平均成绩：77
		姓名：选手D, 平均成绩：81
		姓名：选手E, 平均成绩：69
	*/

	system("pause");
	return EXIT_SUCCESS;
}