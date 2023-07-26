/*
	create for Cxk
	data : 7.25


*/
# define _CRT_SECURE_NO_WARNINGS
# include<iostream>
# include<map>
using namespace std;


/*
	��������
	map<T1, T2> mapTT;//mapĬ�Ϲ��캯��: 
	map(const map &mp);//�������캯��

	��ֵ����
	map& operator=(const map &mp);//���صȺŲ�����
	swap(mp);//����������������

	��С����
	size();//����������Ԫ�ص���Ŀ
	empty();//�ж������Ƿ�Ϊ��

	 �������
	 map.insert(...); //����������Ԫ�أ�����pair<iterator,bool>
	 map<int, string> mapStu;
	 // ��һ�� ͨ��pair�ķ�ʽ�������
	 mapStu.insert(pair<int, string>(3, "С��"));
	 // �ڶ��� ͨ��pair�ķ�ʽ�������
	 mapStu.inset(make_pair(-1, "У��"));
	 // ������ ͨ��value_type�ķ�ʽ�������
	 mapStu.insert(map<int, string>::value_type(1, "С��"));
*/
void printMap(map<int, int>m) {
	// �����鿴map
	cout << "���Դ�ӡMap��" << endl;
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++) {
		cout << "key = " << it->first << ", value = " << (*it).second << endl;
	}
}

void test01() {
	// ��ʼ��
	map<int, int> m;

	// ���뷽ʽ
	// ��һ��
	m.insert(pair<int, int>(1, 10));   // ����pair<iterator,bool>

	// �ڶ���
	m.insert(make_pair(2, 20));

	// ������
	m.insert(map<int, int>::value_type(3, 30));

	// �����֣���ֵ�����Ƽ�
	m[4] = 40;

	// �����鿴map
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++) {
		cout << "key = " << it->first << ", value = " << (*it).second << endl;
	}
	/*
		key = 1, value = 10
		key = 2, value = 20
		key = 3, value = 30
		key = 4, value = 40
	*/

	// ���Ƽ�m[i]��ԭ���ǣ�������mapkeyʱ��������ṩvalue���Զ�������һ��0valueֵ
	// �����ȷ��������ʱʹ��
	cout << m[5] << endl;
	printMap(m);
	/*
		���Դ�ӡMap��
		key = 1, value = 10
		key = 2, value = 20
		key = 3, value = 30
		key = 4, value = 40
		key = 5, value = 0
	*/
}

/*
	2.8.3.5 mapɾ������
		clear();//ɾ������Ԫ��
		erase(pos);//ɾ��pos��������ָ��Ԫ�أ�������һ��Ԫ�صĵ�������
		erase(beg,end);//ɾ������[beg,end)������Ԫ�� ��������һ��Ԫ�صĵ�������
		erase(keyElem);//ɾ��������keyΪkeyElem�Ķ��顣
	2.8.3.5 map���Ҳ���
		find(key);//���Ҽ�key�Ƿ����,�����ڣ����ظü���Ԫ�صĵ�������/�������ڣ�����map.end();
		count(keyElem);//����������keyΪkeyElem�Ķ����������map��˵��Ҫô��0��Ҫô��1����multimap��˵��ֵ���ܴ���1��
		lower_bound(keyElem);//���ص�һ��key<=keyElemԪ�صĵ�������lower_bound(keyElem) �������������в��ҵ�һ����С�ڸ�����ֵ keyElem ��Ԫ�ص�λ�û����λ�á�����һ�ָ�Ч�Ĳ����㷨�������ڸ����������������顣
		upper_bound(keyElem);//���ص�һ��key>keyElemԪ�صĵ�������
		equal_range(keyElem);//����������key��keyElem��ȵ������޵�������������
*/
void test02(){
	map<int, int> m;
	m.insert(pair<int, int>(1, 10));   // ����pair<iterator,bool>				 
	m.insert(make_pair(2, 20));
	m.insert(map<int, int>::value_type(3, 30));
	m[4] = 40;

	// ɾ������
	m.erase(3);	   // ����keyɾ��
	printMap(m);
	/*���Դ�ӡMap��
		key = 1, value = 10
		key = 2, value = 20
		key = 4, value = 40
	*/

	m[3] = 30;
	// ����
	map<int, int>::iterator pos = m.find(3);  // key����
	if (pos != m.end()) {
		cout << "�ҵ��� key Ϊ��" << (*pos).first << ", value Ϊ��" << pos->second << endl;
	}
	// �ҵ��� key Ϊ��3, value Ϊ��30
	
	int num = m.count(4);
	cout << "keyΪ4�Ķ��������" << num << endl;
				// keyΪ4�Ķ��������1

	// 		lower_bound(keyElem);//���ص�һ��key<=keyElemԪ�صĵ�������
	map<int, int>::iterator ret = m.lower_bound(3);		// ���ص�һ��<=3�ĵ�����
	if (ret != m.end()) {
		cout << "�ҵ���lower_bound��keyֵ��" << ret->first << ", valueΪ�� " << ret->second << endl;
	}
	else {
		cout << "δ�ҵ���" << endl;
	}
	// �ҵ���lower_bound��keyֵ��3, valueΪ�� 30

	// upper_bound(keyElem);//���ص�һ��key>keyElemԪ�صĵ�������
	map<int, int>::iterator ret2 = m.upper_bound(3);   //	���ص�һ������3��key������
	if (ret2 != m.end()) {
		cout << "�ҵ���upper_bound��keyֵ��" << ret2->first << ", valueΪ�� " << ret2->second << endl;
	}
	else {
		cout << "δ�ҵ���" << endl;
	}
	// �ҵ���upper_bound��keyֵ��4, valueΪ�� 40

	// equal_range(keyElem);//����������key��keyElem��ȵ������޵�������������
	// ����һ��������pair
	pair<map<int, int>::iterator, map<int, int>::iterator> it2 = m.equal_range(3);  //	���ص�һ������3��key������
	if (it2.first != m.end()) {
		cout << "�ҵ���equal_range�е�keyֵ��" << it2.first->first << ", valueΪ�� " << it2.first->second << endl;
	}
	else {
		cout << "δ�ҵ���" << endl;
	}

}

// �Զ�������   �Ӵ�С
class MyCompare {
public:
	bool operator()(int v1, int v2)const {	  // ��Ҫ�� MyCompare ���еĳ�Ա���� operator() ����Ϊ const��ʹ������� const �������б���ȫ���á�
		return v1 > v2;
	}
};
void test03() {
	map<int, int, MyCompare> m;	// Ĭ�ϴ�С����
	m.insert(pair<int, int>(1, 10));   // ����pair<iterator,bool>				 
	m.insert(make_pair(2, 20));
	m.insert(map<int, int, MyCompare>::value_type(3, 30));
	m[4] = 40;
	// �����鿴map
	cout << "���Դ�ӡMap��" << endl;
	for (map<int, int, MyCompare>::iterator it = m.begin(); it != m.end(); it++) {
		cout << "key = " << it->first << ", value = " << (*it).second << endl;
	};
}


int main() {
	//test01();
	//test02(); //**
	test03();
	system("pause");
	return EXIT_SUCCESS;
}