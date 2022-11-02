#include<iostream>

using namespace std;

/*
	项目需求
	设计立方体类(Cube)，求出立方体的面积( 2*a*b + 2*a*c + 2*b*c )和体积( a * b * c)，
	分别用全局函数和成员函数判断两个立方体是否相等。
*/

// 立方体类
class Cube {
private:
	int m_L;	// 长
	int m_W;	// 宽
	int m_H;	// 高

public:
	// 设置长
	void setL(int l) { m_L = l;	}
	// 获取长
	int getL()const { return m_L;	}  // 成员函数不允许修改
	// 设置宽
	void setW(int w) { m_W = w; }
	// 获取宽
	int getW() { return m_W; }
	// 设置高
	void setH(int h) { m_H = h; }
	// 获取高
	int getH() { return m_H; }
	// 求立方体面积
	void getCubeS() {
		cout << "立方体面积为：" << 2 * m_L * m_W + 2 * m_W * m_H + 2 * m_L * m_H << endl;
	}
	// 求立方体体积
	void getCubeV() {
		cout << "立方体的体积为：" << m_L * m_W * m_H << endl;
	}

	// 2.成员函数判断
	bool compareCubeByClass(Cube &c){
		// 用成员属性判断
		bool ret = m_L == c.getL() &  m_W == c.getW() && m_H == c.getH();
		return ret;
	}
};



// 判断两个立方体是否相等，长宽高都相等,bool类型函数
// 引用传入，直接作业数据本体，不需要临时拆解
// 如果加了const，则成员函数不能调用保护、隐私成员，const下保护乘员不能改变
// const引用可以读取但是不可以被修改引用对象，任何对const引用进行赋值都是不合法的，它适用指向const对象的引用，而非const的引用不适用于指向const对象的引用。
// 1.全局函数判断
bool compareCube( Cube &c1, Cube &c2) {
	if(c1.getL() == c2.getL() && c1.getW() == c2.getW() && c1.getH() == c2.getH()){
		return true;
	}
	return false;
}
// 上述问题解决办法：成员函数也添加const
void func(const Cube & cub){
	cub.getL();		// 正常，getL（）成员函数不允许修改
}



void test01() {
	Cube c1;
	c1.setL(10);
	c1.setH(10);
	c1.setW(10);

	c1.getCubeS();//600
	c1.getCubeV();//1000

	// 通过全局函数判断两个立方体是否相等
	Cube c2;
	c2.setL(10);
	c2.setH(10);
	c2.setW(10);
	// 判断
	bool ret = compareCube(c1, c2);
	if (ret) {
		cout << "全局函数：c1 和 c2 是相等的！！" << endl;
	}
	else {
		cout << "全局函数：c1 和 c2 是不相等的！！" << endl;
	}

	// 通过成员函数判断两个立方体是否相等
	bool ret2 = c1.compareCubeByClass(c2);
	if (ret2) {
		cout << "成员函数：c1 和 c2 是相等的！！" << endl;
	}
	else {
		cout << "成员函数：c1 和 c2 是不相等的！！" << endl;
	}
}
int main() {
	test01();

	system("pause");
	return EXIT_SUCCESS;
}