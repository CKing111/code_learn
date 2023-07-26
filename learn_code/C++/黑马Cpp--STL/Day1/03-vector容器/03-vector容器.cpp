#include<iostream>
#include<vector>

#include<list>

using namespace std;

void test01() {
	vector<int> v;		 // ����һ��vector����
	// ��ͣ��push�鿴�����仯���
	for (int i = 0; i < 10; i++)
	{
		v.push_back(i);
		cout << v.capacity() << " "<<&v[i];
			// endl;  // v.capacity()����������
	}	 //1 2 3 4 6 6 9 9 9 13
	// ����������һ����������
}

/*
	vector���캯��
	vector<T> v; //����ģ��ʵ����ʵ�֣�Ĭ�Ϲ��캯��
	vector(v.begin(), v.end());//��v[begin(), end())�����е�Ԫ�ؿ���������
	vector(n, elem);//���캯����n��elem����������
	vector(const vector &vec);//�������캯����

	//���� ʹ�õڶ������캯�� ���ǿ���...
	int arr[] = {2,3,4,1,9};
	vector<int> v1(arr, arr + sizeof(arr) / sizeof(int));

	��ֵ����
	assign(beg, end);//��[beg, end)�����е����ݿ�����ֵ������
	assign(n, elem);//��n��elem������ֵ������
	vector& operator=(const vector  &vec);//���صȺŲ�����
	swap(vec);// ��vec�뱾���Ԫ�ػ�����
*/
// vector���������
void printVector(vector<int>&v) {
	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}
void test02() {
	vector<int> v1;
	vector<int> v2(10, 100);	 // //���캯����n��elem����������
	printVector(v2);

	vector<int> v3(v2.begin(), v2.end()); // //��v[begin(), end())�����е�Ԫ�ؿ���������
	printVector(v3);

	// ��ֵ����
	vector<int> v4;
	v4.assign(v3.begin(), v3.end());
	printVector(v4);

	int arr[] = { 2,3,4,1,9 };
	vector<int>v5(arr, arr + sizeof(arr) / sizeof(int));   // �ȼ���vector��λ��ַ����

	// swap����
	v4.swap(v5);
	printVector(v4);

}

/*

	size();//����������Ԫ�صĸ���
	empty();//�ж������Ƿ�Ϊ��
	resize(int num);//����ָ�������ĳ���Ϊnum���������䳤������Ĭ��ֵ�����λ�á����������̣���ĩβ�����������ȵ�Ԫ�ر�ɾ����
	resize(int num, elem);//����ָ�������ĳ���Ϊnum���������䳤������elemֵ�����λ�á����������̣���ĩβ����������>�ȵ�Ԫ�ر�ɾ����
	capacity();//����������
	reserve(int len);//����Ԥ��len��Ԫ�س��ȣ�Ԥ��λ�ò���ʼ����Ԫ�ز��ɷ��ʡ�

	vector���ݴ�ȡ����
	at(int idx); //��������idx��ָ�����ݣ����idxԽ�磬�׳�out_of_range�쳣��
	operator[];//��������idx��ָ�����ݣ�Խ��ʱ������ֱ�ӱ���
	front();//���������е�һ������Ԫ��
	back();//�������������һ������Ԫ��

	vector�����ɾ������
	insert(const_iterator pos, int count,ele);//������ָ��λ��pos����count��Ԫ��ele.
	push_back(ele); //β������Ԫ��ele
	pop_back();//ɾ�����һ��Ԫ��
	erase(const_iterator start, const_iterator end);//ɾ����������start��end֮���Ԫ��
	erase(const_iterator pos);//ɾ��������ָ���Ԫ��
	clear();//ɾ������������Ԫ��

*/
void test03() {
	vector<int> v1;
	v1.push_back(10);
	v1.push_back(20);
	v1.push_back(30);
	v1.push_back(40);

	// size()
	cout << "v1��Ԫ�ظ�����" << v1.size() << endl;

	// empty
	if (v1.empty()) {
		cout << "v1Ϊ�գ�" << endl;
	}
	else {
		cout << "v1��Ϊ��" << endl;
	}

	// resize
	// ������
	v1.resize(10);
	printVector(v1);  // 10 20 30 40 0 0 0 0 0 0

	// ������
	v1.resize(3);
	printVector(v1);  // 10 20 30

	// ����resize
	v1.resize(10, 1000);	 // 1000���
	printVector(v1);   // 10 20 30 1000 1000 1000 1000 1000 1000 1000

	// capacity
	cout << "v1��������" << v1.capacity() << endl;	  // v1��������10 ��resize��

	// reserve
	vector<int> v2;
	v2.reserve(20);
	cout << "v2��Ԫ�ظ�����" << v2.size() << ", v2��������" << v2.capacity() << ", v2�Ƿ�Ϊ�գ�" << v2.empty() << endl;

	// front()
	printVector(v1);
	cout << "v1�ĵ�һ��Ԫ���ǣ�" << v1.front() << endl;
	// v1�ĵ�һ��Ԫ���ǣ�10

	// back
	cout << "v1�����һ��Ԫ���ǣ�" << v1.back() << endl;
	// v1�����һ��Ԫ���ǣ�1000

	// insert����
	v1.insert(v1.begin(), 3, 101);
	cout << "������v1:";
	printVector(v1);
	// ������v1:101 101 101 10 20 30 1000 1000 1000 1000 1000 1000 1000

	// push_back
	v1.push_back(202);
	cout << "v1β���";
	printVector(v1);  
	// v1β���101 101 101 10 20 30 1000 1000 1000 1000 1000 1000 1000 202

	// pop_back
	v1.pop_back();
	cout << "v1βɾ��";
	printVector(v1);
	// v1βɾ��101 101 101 10 20 30 1000 1000 1000 1000 1000 1000 1000

	// ɾ��
	// ע�⣺1. erase������ǵ����������ɾ��ָ��λ�ã�����ɾ���м�ɾ������Ϊ 1 �� 2 ��Ԫ��(����ҿ�):v.begin() + 1, v.begin() + 3
	//       2. erase ����ɾ��Ԫ�غ󣬺����Ԫ�ػ���ǰ�ƶ����ȱ����������Ĵ�С���С��
	v1.erase(v1.begin()); // ɾ����һ��
	printVector(v1);  // 101 101 10 20 30 1000 1000 1000 1000 1000 1000 1000
	v1.erase(v1.begin(), v1.end()); // ����������ȼ���v1.clear()
	printVector(v1);  // " "

}


// Ӧ�ü���
// 1.����reserve���ӳ���Ч�ʣ�
/*
		�ʣ�reserve��resize������?
��:  reserve������Ԥ���ռ䣬���ڿռ��ڲ���������Ԫ�ض���������û������µĶ���֮ǰ���������������ڵ�Ԫ��.
    resize�Ǹı������Ĵ�С�����ڴ���������ˣ������������֮�󣬾Ϳ������������ڵĶ�����.
  ����reserve���ӳ���Ч�ʣ�
	 ��Ҫע����ǣ�ʹ�� reserve ����ֻ��Ԥ�����㹻���ڴ�ռ䣬������ı������Ĵ�С��
	 ��ˣ���ʹ�� reserve �����������Ĵ�С��Ȼ��0��ֻ�е��������в���Ԫ��ʱ�������Ĵ�С�Ż����ӡ�
*/

void test04() {
	// ��ʹ��reserve
	vector<int> v1;
	int *p1 = NULL;
	int count1 = 0;   //��¼vector�����仯����
	for (int i = 0; i < 100000; i++) {
		v1.push_back(i);
		// &v1[0] ��ʾ���� v1 �е�һ��Ԫ�صĵ�ַ����������Ԫ�ص���ʼ��ַ��
		if (p1 != &v1[0]) {		// �Ƚϸõ�ַ���²���Ԫ�صĵ�ַ�Ƿ�������ж������Ƿ����·����ڴ档
			     				// ������������·����ڴ棬&v1[0] ��ֵ�ͻᷢ���仯����ʱ p2 �Ͳ����� &v1[0]
			p1 = &v1[0];
			count1++;
		}
	}
	cout << "��ʹ��reserve��vector����10���ֵ�������任������" << count1 << endl;
	// ��ʹ��reserve��vector����10���ֵ�������任������30

	// ʹ��reserve
	vector<int> v2;
	v2.reserve(100000);
	cout << "v2��ʼ����capacity��С��" << v2.capacity() <<", v2��size��С��"<<v2.size()<< endl;
	// v2��ʼ����capacity��С��100000, v2��size��С��0
	int *p2 = NULL;
	int count2 = 0;   //��¼vector�����仯����
	for (int i = 0; i < 100000; i++) {
		v2.push_back(i);
		if (p2 != &v2[0]) {		
			p2 = &v2[0];
			count2++;
		}
	}
	cout << "ʹ��reserve��vector����10���ֵ�������任������" << count2 << endl;
	// ʹ��reserve��vector����10���ֵ�������任������1
	cout << "v2ʹ�ú��capacity��С��" << v2.capacity() << ", v2��size��С��" << v2.size() << endl;
	// v2ʹ�ú��capacity��С��100000, v2��size��С��100000

	// ʹ��reserve��vector����10���ֵ�������任������1
	// v2 �ĳ�ʼ����Ϊ100000��Ԫ�أ�����ѭ�����������в���100000������Ԫ�ء�
	// ��֮ǰ�Ĵ��벻ͬ���ǣ�����ʹ���� reserve ���������� v2 ��ʼʱ�Ѿ��������㹻���ڴ�ռ䣬������������в���Ԫ��ʱ��
	// ��������Ҫ���·����ڴ棬Ҳ�Ͳ��ᷢ�������仯����ˣ�count2 ������ֵӦ��Ϊ1��
	// ͨ��ʹ�� reserve ����Ԥ���㹻�����������Ա����������в���Ԫ��ʱƵ�������·����ڴ棬�Ӷ���߳�������ܡ�
}


// 2. ����swap�����ڴ�
void test05() {
	vector<int> v;

	for (int i = 0; i < 100000; i++) {
		v.push_back(i);
	}
	cout << "v��������" << v.capacity() << endl;	// v��������138255
	cout << "v�Ĵ�С��" << v.size() << endl;		// v�Ĵ�С��100000

	v.resize(3);
	cout << "resize��v��������" << v.capacity() << endl;	// resize��v��������138255
	cout << "resize��v�Ĵ�С��" << v.size() << endl;		// resize��v�Ĵ�С��3

	// ������������˷�
	// �����ڴ�
	// vector<int>(v) ���Ǵ�����һ�������� vector<int> �������������� v �����Ŀ����������������û�б���ֵ���κα������������һ����ʱ����Ҳ��Ϊ��ֵ����
	vector<int>(v).swap(v);	  // // ʹ����ʱ vector ���󽻻��������ݣ��ͷŶ�����ڴ�ռ�
	cout << "swap��v��������" << v.capacity() << endl;	// swap��v��������3
	cout << "swap��v�Ĵ�С��" << v.size() << endl;		// swap��v�Ĵ�С��3

	/*
			����ʹ�������� vector<int> ���������� v ���������ݣ�Ȼ����� swap ��������ʱ����� v ���������ݽ�����
			������ʱ������ڴ�ռ��С��ʵ����Ҫ�Ĵ�С��ͬ����˽�����v �������ڴ�ռ�Ҳ�ᱻ��С��ʵ����Ҫ�Ĵ�С��
			�Ӷ��ͷŶ�����ڴ�ռ䡣

			vector<int>(v) ������һ�������� vector<int> �������������� v �����Ŀ����������������û�б���ֵ���κα�����
			�������һ����ʱ����Ҳ��Ϊ��ֵ����

			�ڵ��� swap ����ʱ�����ǽ������ʱ������Ϊ�������ݸ��� swap �������������ݺ� v ���������ݽ����˽�����
			���������������û�б�ʹ�ã�������ڽ�����ͻᱻ���٣���Ҳ�����������һ���ص㡣

			��Ҫע����ǣ��������������������һ�����ʽ�������������ڽ����󣬶���ͻᱻ���١���ʹ����������ʱ����Ҫע�������������ڣ�
			�����������ָ������⡣
	*/
}

// 3. �������
void test06() {
	vector<int> v;
	v.push_back(10);
	v.push_back(20);
	v.push_back(30);
	v.push_back(40);

	cout << "���������";
	printVector(v);	   // ���������10 20 30 40
	cout << endl;
	cout<< "���������";	  // ���������40 30 20 10
	for (vector<int>::reverse_iterator it = v.rbegin(); it != v.rend(); it++) {		// ����ӿ�
		cout << *it << " ";
	}

	// vector�����ĵ�������������ʵ�����
	// ����жϵ������Ƿ�֧��������ʣ�

	vector<int>::iterator itBegin = v.begin();  // ����������
	itBegin = itBegin + 1;    // ����﷨֧�֣������֧���������
	itBegin++;				// ֧��˫��
	itBegin--;

	list<int> l;	// list����
	l.push_back(10);
	l.push_back(20);
	l.push_back(30);

	list<int>::iterator it2 = l.begin();
	it2++;			  // ֧��˫��
	it2--;
	//it2 = it2 + 1;	 // ��֧���������	 ����Ϊ�洢��������ʵ�ָ���


}
int main() {
	//test01();
	//test02();
	//test03();	// **
	//test04();	// *								 
	//test05();	// **
	test06();

	system("pause");
	return EXIT_SUCCESS;
}

