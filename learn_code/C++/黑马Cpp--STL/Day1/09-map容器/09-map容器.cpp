/*
	create for Cxk
	data : 7.25


*/
# define _CRT_SECURE_NO_WARNINGS
# include<iostream>
# include<map>
using namespace std;


/*
	构造容器
	map<T1, T2> mapTT;//map默认构造函数: 
	map(const map &mp);//拷贝构造函数

	赋值操作
	map& operator=(const map &mp);//重载等号操作符
	swap(mp);//交换两个集合容器

	大小操作
	size();//返回容器中元素的数目
	empty();//判断容器是否为空

	 插入操作
	 map.insert(...); //往容器插入元素，返回pair<iterator,bool>
	 map<int, string> mapStu;
	 // 第一种 通过pair的方式插入对象
	 mapStu.insert(pair<int, string>(3, "小张"));
	 // 第二种 通过pair的方式插入对象
	 mapStu.inset(make_pair(-1, "校长"));
	 // 第三种 通过value_type的方式插入对象
	 mapStu.insert(map<int, string>::value_type(1, "小李"));
*/
void printMap(map<int, int>m) {
	// 遍历查看map
	cout << "测试打印Map：" << endl;
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++) {
		cout << "key = " << it->first << ", value = " << (*it).second << endl;
	}
}

void test01() {
	// 初始化
	map<int, int> m;

	// 插入方式
	// 第一种
	m.insert(pair<int, int>(1, 10));   // 返回pair<iterator,bool>

	// 第二种
	m.insert(make_pair(2, 20));

	// 第三种
	m.insert(map<int, int>::value_type(3, 30));

	// 第四种，赋值，不推荐
	m[4] = 40;

	// 遍历查看map
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++) {
		cout << "key = " << it->first << ", value = " << (*it).second << endl;
	}
	/*
		key = 1, value = 10
		key = 2, value = 20
		key = 3, value = 30
		key = 4, value = 40
	*/

	// 不推荐m[i]的原因是，在声明mapkey时，如果不提供value会自动声明出一个0value值
	// 如果明确索引存在时使用
	cout << m[5] << endl;
	printMap(m);
	/*
		测试打印Map：
		key = 1, value = 10
		key = 2, value = 20
		key = 3, value = 30
		key = 4, value = 40
		key = 5, value = 0
	*/
}

/*
	2.8.3.5 map删除操作
		clear();//删除所有元素
		erase(pos);//删除pos迭代器所指的元素，返回下一个元素的迭代器。
		erase(beg,end);//删除区间[beg,end)的所有元素 ，返回下一个元素的迭代器。
		erase(keyElem);//删除容器中key为keyElem的对组。
	2.8.3.5 map查找操作
		find(key);//查找键key是否存在,若存在，返回该键的元素的迭代器；/若不存在，返回map.end();
		count(keyElem);//返回容器中key为keyElem的对组个数。对map来说，要么是0，要么是1。对multimap来说，值可能大于1。
		lower_bound(keyElem);//返回第一个key<=keyElem元素的迭代器。lower_bound(keyElem) 是在有序序列中查找第一个不小于给定键值 keyElem 的元素的位置或插入位置。它是一种高效的查找算法，适用于各种有序容器和数组。
		upper_bound(keyElem);//返回第一个key>keyElem元素的迭代器。
		equal_range(keyElem);//返回容器中key与keyElem相等的上下限的两个迭代器。
*/
void test02(){
	map<int, int> m;
	m.insert(pair<int, int>(1, 10));   // 返回pair<iterator,bool>				 
	m.insert(make_pair(2, 20));
	m.insert(map<int, int>::value_type(3, 30));
	m[4] = 40;

	// 删除操作
	m.erase(3);	   // 按照key删除
	printMap(m);
	/*测试打印Map：
		key = 1, value = 10
		key = 2, value = 20
		key = 4, value = 40
	*/

	m[3] = 30;
	// 查找
	map<int, int>::iterator pos = m.find(3);  // key查找
	if (pos != m.end()) {
		cout << "找到了 key 为：" << (*pos).first << ", value 为：" << pos->second << endl;
	}
	// 找到了 key 为：3, value 为：30
	
	int num = m.count(4);
	cout << "key为4的队组个数：" << num << endl;
				// key为4的队组个数：1

	// 		lower_bound(keyElem);//返回第一个key<=keyElem元素的迭代器。
	map<int, int>::iterator ret = m.lower_bound(3);		// 返回第一个<=3的迭代器
	if (ret != m.end()) {
		cout << "找到了lower_bound的key值：" << ret->first << ", value为： " << ret->second << endl;
	}
	else {
		cout << "未找到！" << endl;
	}
	// 找到了lower_bound的key值：3, value为： 30

	// upper_bound(keyElem);//返回第一个key>keyElem元素的迭代器。
	map<int, int>::iterator ret2 = m.upper_bound(3);   //	返回第一个大于3的key迭代器
	if (ret2 != m.end()) {
		cout << "找到了upper_bound的key值：" << ret2->first << ", value为： " << ret2->second << endl;
	}
	else {
		cout << "未找到！" << endl;
	}
	// 找到了upper_bound的key值：4, value为： 40

	// equal_range(keyElem);//返回容器中key与keyElem相等的上下限的两个迭代器。
	// 传入一个迭代器pair
	pair<map<int, int>::iterator, map<int, int>::iterator> it2 = m.equal_range(3);  //	返回第一个大于3的key迭代器
	if (it2.first != m.end()) {
		cout << "找到了equal_range中的key值：" << it2.first->first << ", value为： " << it2.first->second << endl;
	}
	else {
		cout << "未找到！" << endl;
	}

}

// 自定义排序   从大到小
class MyCompare {
public:
	bool operator()(int v1, int v2)const {	  // 需要将 MyCompare 类中的成员函数 operator() 声明为 const，使其可以在 const 上下文中被安全调用。
		return v1 > v2;
	}
};
void test03() {
	map<int, int, MyCompare> m;	// 默认从小到大
	m.insert(pair<int, int>(1, 10));   // 返回pair<iterator,bool>				 
	m.insert(make_pair(2, 20));
	m.insert(map<int, int, MyCompare>::value_type(3, 30));
	m[4] = 40;
	// 遍历查看map
	cout << "测试打印Map：" << endl;
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