#include <iostream>;

using namespace std;

// 房屋类
class Building {
	// 友元函数
	// 有一个全局函数，作为本类的友元函数，可以访问私有内容；
	friend void goodGay(Building& b);
public:
	// 构造函数，初始化参数
	Building(){
		this->m_SittingRoom = "客厅";
		this->m_BedRoom = "卧室";
	}

	// 
	string m_SittingRoom;	//  客厅，公开

private:
	string m_BedRoom;		// 卧室，私有
};

// 全局函数，可作为友元函数
void goodGay(Building& b) {
	cout << "好基友正在访问：" << b.m_SittingRoom << endl;

	cout << "好基友正在访问：" << b.m_BedRoom<< endl; // 失败，设置友元friend后成功

}

void test01(){
	Building b;
	goodGay(b);
}
int main() {
	test01();
	system( "pause" );
	return EXIT_SUCCESS;
}