/*
	create by Cxk
	data : 2023.7.25
	createPerson(vector<Person>& v)��������ѡ�֣�����1����������

*/

#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<deque>

#include<ctime>
using namespace std;

/*
		��ĿҪ��
		������ѡ�֣�ѡ��ABCDE,10λ��ί�ֱ��ÿһ��ѡ�ִ�֣�ȥ����߷֣�ȥ����ί��ͷ֣�ȡƽ���֡�
		1. ��������ѡ�֣��ŵ�vector�����У�
		2. ����vector������ȡ��ÿһ��ѡ�֣�ִ�з���ѭ�������԰�10�����ִ�ִ浽deque�����У�
		3. sort�㷨��deque�����з�������pop_back pop_frontȥ����߷ֺ���ͷ֣�
		4. deque�����������ۼӷ������ۼӷ���/size()����ƽ���֣�
		5. ��person.score = ƽ���֣�
*/

// ѡ����
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
	string m_Name;	// ����
	int m_Score;	// ƽ����
};
void test01() {

}

void createPerson(vector<Person>&v) {
	// ѡ��������
	string nameSeed = "ABCDE";
	for (int i = 0; i < 5; i++) {
		string name = "ѡ��";
		name += nameSeed[i];

		int score = 0;
		Person p(name, score);
		v.push_back(p);     // ����1�ڵ�
	}

}
void printVector(vector<Person> v) {
	// ���Խڵ�1
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++) {
		cout << "������" << (*it).getName() << ", �ɼ���" << (*it).getScore() << endl;
		// error:û������Щ������ƥ��� "<<" �����	04 - ��ί��ְ���	d : \code\TrainingCode\����Cpp--STL\Day1\04 - ��ί��ְ���\04 - ��ί��ְ���.cpp	61
		// û������string ͷ�ļ�
	}
}
void printVectorAvg(vector<Person> v) {
	// ���Խڵ�1
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++) {
		cout << "������" << (*it).getName() << ", ƽ���ɼ���" << (*it).getScore() << endl;
		// error:û������Щ������ƥ��� "<<" �����	04 - ��ί��ְ���	d : \code\TrainingCode\����Cpp--STL\Day1\04 - ��ί��ְ���\04 - ��ί��ְ���.cpp	61
		// û������string ͷ�ļ�
	}
}
void printVector(deque<int> v) {
	// ���Խڵ�1		
	for (deque<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it<<" ";
	}
	cout << endl;
}
void printVector(vector<int> v) {
	// ���Խڵ�1		
	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

// ��ֺ���
void setScore(vector<Person>&v) {
	// ѭ��ѡ��
	for (vector<Person>::iterator it = v.begin(); it != v.end(); it++) {
		// ��10����ί��֣������������
		deque<int>d;
		for (int i = 0; i < 10; i++) {
			int score = rand() % 41 + 60;    // ������䣺60~99
			// %��ȡģ���������ʾ����ߵ��������ұߵ�����Ȼ��ȡ��������ˣ�rand() % 40�Ľ����һ������0��39֮���������
			// ��ˣ�score = rand() % 40 + 60�Ľ�����ǽ�0��39֮��������������60, �õ�һ��60��100֮������������
			d.push_back(score);
		}
		//cout << "ѡ�֣� " << it->getName() << "�Ĵ�ֽ����" << endl;
		//printVector(d);// ���Ե�2������
		//cout << endl;

		// ����
		sort(d.begin(), d.end());		   // Ĭ�ϴ�С����
		//printVector(d);// ���Ե�2������	
        /*
				ѡ�֣� ѡ��A�Ĵ�ֽ����
				60 77 80 74 82 81 99 62 85 88
				60 62 74 77 80 81 82 85 88 99
				ѡ�֣� ѡ��B�Ĵ�ֽ����
				66 79 94 77 99 100 62 71 90 84
				62 66 71 77 79 84 90 94 99 100
				ѡ�֣� ѡ��C�Ĵ�ֽ����
				61 68 67 90 65 60 97 80 98 70
				60 61 65 67 68 70 80 90 97 98
				ѡ�֣� ѡ��D�Ĵ�ֽ����
				95 97 71 77 84 87 61 78 80 73
				61 71 73 77 78 80 84 87 95 97
				ѡ�֣� ѡ��E�Ĵ�ֽ����
				63 91 99 94 62 91 72 63 64 81
				62 63 63 64 72 81 91 91 94 99
		 */

		// ȥ����߷ֺ���ͷ�
		// ��߷�
		d.pop_back();
		// ��ͷ�
		d.pop_front();

		// ��ȡ�ܷ�
		int sum = 0;
		for (deque<int>::iterator it = d.begin(); it != d.end(); it++) {
			sum += *it;
		}

		// ��ȡƽ����
		int avg = sum / d.size();
		
		// ������ֵ
		it->setScore(avg);
	}
}
int main() {

	// �������������
	//srand((unsigned int)time(NULL));
	srand((unsigned int)10);
		// srand()���������������������������ֵ��
		// time(NULL)��һ��ϵͳ���������ص�ǰʱ���ʱ���(����Ϊ��λ)��
		// �����ʱ���ת��Ϊ�޷������ͺ���Ϊ����ֵ����srand()���������Ա�֤ÿ�����г���ʱ���ɵ���������ж���ͬ��
	// 1. �����������
	vector<Person> v;
	// 2. ����5��ѡ��
	createPerson(v);
	//printVector(v);	// ����1

	// 3. ���
	setScore(v);

	// 4. ��ʾ�÷�
	printVectorAvg(v);
	/*
		������ѡ��A, ƽ���ɼ���74
		������ѡ��B, ƽ���ɼ���74
		������ѡ��C, ƽ���ɼ���77
		������ѡ��D, ƽ���ɼ���81
		������ѡ��E, ƽ���ɼ���69
	*/

	system("pause");
	return EXIT_SUCCESS;
}