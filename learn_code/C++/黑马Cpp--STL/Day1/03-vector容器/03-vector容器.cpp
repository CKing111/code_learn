#include<iostream>
#include<vector>

#include<list>

using namespace std;

void test01() {
	vector<int> v;		 // 声明一个vector容器
	// 不停的push查看容量变化情况
	for (int i = 0; i < 10; i++)
	{
		v.push_back(i);
		cout << v.capacity() << " "<<&v[i];
			// endl;  // v.capacity()容器的容量
	}	 //1 2 3 4 6 6 9 9 9 13
	// 容量增加有一定提升规律
}

/*
	vector构造函数
	vector<T> v; //采用模板实现类实现，默认构造函数
	vector(v.begin(), v.end());//将v[begin(), end())区间中的元素拷贝给本身。
	vector(n, elem);//构造函数将n个elem拷贝给本身。
	vector(const vector &vec);//拷贝构造函数。

	//例子 使用第二个构造函数 我们可以...
	int arr[] = {2,3,4,1,9};
	vector<int> v1(arr, arr + sizeof(arr) / sizeof(int));

	赋值操作
	assign(beg, end);//将[beg, end)区间中的数据拷贝赋值给本身。
	assign(n, elem);//将n个elem拷贝赋值给本身。
	vector& operator=(const vector  &vec);//重载等号操作符
	swap(vec);// 将vec与本身的元素互换。
*/
// vector输出迭代器
void printVector(vector<int>&v) {
	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}
void test02() {
	vector<int> v1;
	vector<int> v2(10, 100);	 // //构造函数将n个elem拷贝给本身。
	printVector(v2);

	vector<int> v3(v2.begin(), v2.end()); // //将v[begin(), end())区间中的元素拷贝给本身。
	printVector(v3);

	// 赋值操作
	vector<int> v4;
	v4.assign(v3.begin(), v3.end());
	printVector(v4);

	int arr[] = { 2,3,4,1,9 };
	vector<int>v5(arr, arr + sizeof(arr) / sizeof(int));   // 等价于vector首位地址声明

	// swap交换
	v4.swap(v5);
	printVector(v4);

}

/*

	size();//返回容器中元素的个数
	empty();//判断容器是否为空
	resize(int num);//重新指定容器的长度为num，若容器变长，则以默认值填充新位置。如果容器变短，则末尾超出容器长度的元素被删除。
	resize(int num, elem);//重新指定容器的长度为num，若容器变长，则以elem值填充新位置。如果容器变短，则末尾超出容器长>度的元素被删除。
	capacity();//容器的容量
	reserve(int len);//容器预留len个元素长度，预留位置不初始化，元素不可访问。

	vector数据存取操作
	at(int idx); //返回索引idx所指的数据，如果idx越界，抛出out_of_range异常。
	operator[];//返回索引idx所指的数据，越界时，运行直接报错
	front();//返回容器中第一个数据元素
	back();//返回容器中最后一个数据元素

	vector插入和删除操作
	insert(const_iterator pos, int count,ele);//迭代器指向位置pos插入count个元素ele.
	push_back(ele); //尾部插入元素ele
	pop_back();//删除最后一个元素
	erase(const_iterator start, const_iterator end);//删除迭代器从start到end之间的元素
	erase(const_iterator pos);//删除迭代器指向的元素
	clear();//删除容器中所有元素

*/
void test03() {
	vector<int> v1;
	v1.push_back(10);
	v1.push_back(20);
	v1.push_back(30);
	v1.push_back(40);

	// size()
	cout << "v1的元素个数：" << v1.size() << endl;

	// empty
	if (v1.empty()) {
		cout << "v1为空！" << endl;
	}
	else {
		cout << "v1不为空" << endl;
	}

	// resize
	// 超容量
	v1.resize(10);
	printVector(v1);  // 10 20 30 40 0 0 0 0 0 0

	// 少容量
	v1.resize(3);
	printVector(v1);  // 10 20 30

	// 重载resize
	v1.resize(10, 1000);	 // 1000填充
	printVector(v1);   // 10 20 30 1000 1000 1000 1000 1000 1000 1000

	// capacity
	cout << "v1的容量：" << v1.capacity() << endl;	  // v1的容量：10 （resize后）

	// reserve
	vector<int> v2;
	v2.reserve(20);
	cout << "v2的元素个数：" << v2.size() << ", v2的容量：" << v2.capacity() << ", v2是否为空：" << v2.empty() << endl;

	// front()
	printVector(v1);
	cout << "v1的第一个元素是：" << v1.front() << endl;
	// v1的第一个元素是：10

	// back
	cout << "v1的最后一个元素是：" << v1.back() << endl;
	// v1的最后一个元素是：1000

	// insert插入
	v1.insert(v1.begin(), 3, 101);
	cout << "插入后的v1:";
	printVector(v1);
	// 插入后的v1:101 101 101 10 20 30 1000 1000 1000 1000 1000 1000 1000

	// push_back
	v1.push_back(202);
	cout << "v1尾插后：";
	printVector(v1);  
	// v1尾插后：101 101 101 10 20 30 1000 1000 1000 1000 1000 1000 1000 202

	// pop_back
	v1.pop_back();
	cout << "v1尾删后：";
	printVector(v1);
	// v1尾删后：101 101 101 10 20 30 1000 1000 1000 1000 1000 1000 1000

	// 删除
	// 注意：1. erase输入的是迭代器，如果删除指定位置，比如删除中间删除索引为 1 和 2 的元素(左闭右开):v.begin() + 1, v.begin() + 3
	//       2. erase 方法删除元素后，后面的元素会向前移动填补空缺，因此容器的大小会减小。
	v1.erase(v1.begin()); // 删除第一个
	printVector(v1);  // 101 101 10 20 30 1000 1000 1000 1000 1000 1000 1000
	v1.erase(v1.begin(), v1.end()); // 清空容器，等价于v1.clear()
	printVector(v1);  // " "

}


// 应用技巧
// 1.巧用reserve增加程序效率？
/*
		问：reserve和resize的区别?
答:  reserve是容器预留空间，但在空间内不真正创建元素对象，所以在没有添加新的对象之前，不能引用容器内的元素.
    resize是改变容器的大小，且在创建对象，因此，调用这个函数之后，就可以引用容器内的对象了.
  巧用reserve增加程序效率？
	 需要注意的是，使用 reserve 方法只是预留了足够的内存空间，并不会改变容器的大小。
	 因此，在使用 reserve 方法后，容器的大小仍然是0，只有当向容器中插入元素时，容器的大小才会增加。
*/

void test04() {
	// 不使用reserve
	vector<int> v1;
	int *p1 = NULL;
	int count1 = 0;   //记录vector容量变化次数
	for (int i = 0; i < 100000; i++) {
		v1.push_back(i);
		// &v1[0] 表示容器 v1 中第一个元素的地址，即容器中元素的起始地址。
		if (p1 != &v1[0]) {		// 比较该地址和新插入元素的地址是否相等来判断容器是否重新分配内存。
			     				// 如果发生了重新分配内存，&v1[0] 的值就会发生变化，此时 p2 就不等于 &v1[0]
			p1 = &v1[0];
			count1++;
		}
	}
	cout << "不使用reserve，vector存入10万个值，容量变换次数：" << count1 << endl;
	// 不使用reserve，vector存入10万个值，容量变换次数：30

	// 使用reserve
	vector<int> v2;
	v2.reserve(100000);
	cout << "v2初始化的capacity大小：" << v2.capacity() <<", v2的size大小："<<v2.size()<< endl;
	// v2初始化的capacity大小：100000, v2的size大小：0
	int *p2 = NULL;
	int count2 = 0;   //记录vector容量变化次数
	for (int i = 0; i < 100000; i++) {
		v2.push_back(i);
		if (p2 != &v2[0]) {		
			p2 = &v2[0];
			count2++;
		}
	}
	cout << "使用reserve，vector存入10万个值，容量变换次数：" << count2 << endl;
	// 使用reserve，vector存入10万个值，容量变换次数：1
	cout << "v2使用后的capacity大小：" << v2.capacity() << ", v2的size大小：" << v2.size() << endl;
	// v2使用后的capacity大小：100000, v2的size大小：100000

	// 使用reserve，vector存入10万个值，容量变换次数：1
	// v2 的初始容量为100000个元素，并在循环中向容器中插入100000个整数元素。
	// 与之前的代码不同的是，由于使用了 reserve 方法，容器 v2 初始时已经分配了足够的内存空间，因此在向容器中插入元素时，
	// 容器不需要重新分配内存，也就不会发生容量变化。因此，count2 变量的值应该为1。
	// 通过使用 reserve 方法预留足够的容量，可以避免向容器中插入元素时频繁地重新分配内存，从而提高程序的性能。
}


// 2. 巧用swap收缩内存
void test05() {
	vector<int> v;

	for (int i = 0; i < 100000; i++) {
		v.push_back(i);
	}
	cout << "v的容量：" << v.capacity() << endl;	// v的容量：138255
	cout << "v的大小：" << v.size() << endl;		// v的大小：100000

	v.resize(3);
	cout << "resize后，v的容量：" << v.capacity() << endl;	// resize后，v的容量：138255
	cout << "resize后，v的大小：" << v.size() << endl;		// resize后，v的大小：3

	// 造成了容量的浪费
	// 收缩内存
	// vector<int>(v) 就是创建了一个匿名的 vector<int> 对象，它的内容是 v 容器的拷贝。这个匿名对象没有被赋值给任何变量，因此它是一个临时对象，也称为右值对象。
	vector<int>(v).swap(v);	  // // 使用临时 vector 对象交换容器内容，释放多余的内存空间
	cout << "swap后，v的容量：" << v.capacity() << endl;	// swap后，v的容量：3
	cout << "swap后，v的大小：" << v.size() << endl;		// swap后，v的大小：3

	/*
			首先使用匿名的 vector<int> 对象来复制 v 容器的内容，然后调用 swap 函数将临时对象和 v 容器的内容交换。
			由于临时对象的内存空间大小与实际需要的大小相同，因此交换后，v 容器的内存空间也会被缩小到实际需要的大小，
			从而释放多余的内存空间。

			vector<int>(v) 创建了一个匿名的 vector<int> 对象，它的内容是 v 容器的拷贝。这个匿名对象没有被赋值给任何变量，
			因此它是一个临时对象，也称为右值对象。

			在调用 swap 函数时，我们将这个临时对象作为参数传递给了 swap 函数，它的内容和 v 容器的内容进行了交换。
			由于这个匿名对象没有被使用，因此它在交换后就会被销毁，这也是匿名对象的一个特点。

			需要注意的是，匿名对象的生命周期是一个表达式，它的生命周期结束后，对象就会被销毁。在使用匿名对象时，需要注意它的生命周期，
			避免出现悬空指针等问题。
	*/
}

// 3. 逆序遍历
void test06() {
	vector<int> v;
	v.push_back(10);
	v.push_back(20);
	v.push_back(30);
	v.push_back(40);

	cout << "正序遍历：";
	printVector(v);	   // 正序遍历：10 20 30 40
	cout << endl;
	cout<< "逆序遍历：";	  // 逆序遍历：40 30 20 10
	for (vector<int>::reverse_iterator it = v.rbegin(); it != v.rend(); it++) {		// 逆序接口
		cout << *it << " ";
	}

	// vector容器的迭代器，随机访问迭代器
	// 如何判断迭代器是否支持随机访问？

	vector<int>::iterator itBegin = v.begin();  // 声明迭代器
	itBegin = itBegin + 1;    // 如果语法支持，则表明支持随机访问
	itBegin++;				// 支持双向
	itBegin--;

	list<int> l;	// list容器
	l.push_back(10);
	l.push_back(20);
	l.push_back(30);

	list<int>::iterator it2 = l.begin();
	it2++;			  // 支持双向
	it2--;
	//it2 = it2 + 1;	 // 不支持随机访问	 ，因为存储不连续，实现复杂


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

