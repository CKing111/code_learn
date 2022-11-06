#include<iostream>

using namespace std;

/*
	单例模式：就是为了创建类中的对象，并且保证只有一个对象实例（不能构造、拷贝一个新的相同类实例）
				单例模式通常不会自动释放空间，默认只有一个指针占空间小
	步骤：
		将构造函数和拷贝构造私有化；
		内部维护一个对象指针；
		私有化唯一的对象指针；
		对外提供getInstangce方法来访问这个对象指针；
		保证类中只能实例化一个对象；

*/



// 创建主席类
class ChairMan {

private:	
	// 私有化，构造函数
	ChairMan() {
		cout << "调用私有空间ChairMan构造函数！" << endl;
	}
	// 需要保护共享变量，放在private中
	static ChairMan* singeMan;		// 声明静态变量维护一个类指针，共享数据
	// 拷贝构造私有化
	ChairMan(const ChairMan& c) {}


public:		// 共享空间
	// 提供静态成员get方法，访问共享的类对象
	// 该方法是的，参数无法被定义为NULL
	static ChairMan* getInstance() {
		return singeMan;
	}
};
// 类外初始化
ChairMan* ChairMan::singeMan = new ChairMan;

void test01() {
	//ChairMan c1;
	//ChairMan* c2 = new ChairMan;
	//ChairMan* c3 = new ChairMan;

	// 放入私有空间后无法使用
	//ChairMan::singeMan;		// 共享数据
	//// 指针获取
	//ChairMan* cm = ChairMan::singeMan;
	//ChairMan* cm2 = ChairMan::singeMan;

	// 使用get方法调用
	//ChairMan::getInstance() = NULL;			// 失败，get方法没有权限修改指针源
	ChairMan* cm1 = ChairMan::getInstance();	// 成功，获取指针
	ChairMan* cm2 = ChairMan::getInstance();
	if (cm1 == cm2) {
		cout << "cm1 与 cm2 相同！" << endl;
	}
	else {
		cout << "cm1 与 cm2 不相同！" << endl;
	}

	// 当调用默认拷贝函数时，两个对象不是同一个共享数据，cm2 != cm3
	//ChairMan* cm3 = new ChairMan(*cm2);
	//if (cm2 == cm3) {
	//	cout << "cm3 与 cm2 相同！" << endl;
	//}
	//else {
	//	cout << "cm3 与 cm2 不相同！" << endl;
	//}

}

int main() {
	cout << "Main函数调用！" << endl;

	test01();

	system("pause");
	return EXIT_SUCCESS;
}