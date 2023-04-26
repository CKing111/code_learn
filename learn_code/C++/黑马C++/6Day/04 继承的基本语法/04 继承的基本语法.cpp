#include<iostream>

using namespace std;

// 传统方法会出现大量重复代码
// 
//// 游戏页面
//class GamePage {
//public:
//	void header() {
//		cout << "公共头部" << endl;
//	}
//	void footer() {
//		cout<<"公共底部" << endl;
//	}
//	void leftList() {
//		cout << "公共的左侧列表" << endl;
//	}
//	void content() {
//		cout << "LOL" << endl;
//	}
//};
//
//// 新闻页面
//class NewPage {
//public:
//	void header() {
//		cout << "公共头部" << endl;
//	}
//	void footer() {
//		cout << "公共底部" << endl;
//	}
//	void leftList() {
//		cout << "公共的左侧列表" << endl;
//	}
//	void content() {
//		cout << "NEWs" << endl;
//	}
//};

// 使用继承，减少重复代码的出现
// BasePage:基类、父类
// GamePage\NewsPage: 派生类、子类 
// 继承方式：
// 公共继承：class 子类 ：public 父类{}--------不可访问父类私有，其他不变
// 保护继承：class 子类 ：protected 父类{}-----不可访问父类私有，其他变保护
// 私有继承：class 子类 ：private 父类{}-------不可访问父类私有，其他变私有
class BasePage {
public:
	void header() {
		cout << "公共头部" << endl;
	}
	void footer() {
		cout << "公共底部" << endl;
	}
	void leftList() {
		cout << "公共的左侧列表" << endl;
	}
};
// 新闻类继承
class NewsPage :public BasePage {
public:
	void content() {
		cout << "两会报道" << endl;
	}
};
// 游戏类继承
class GamePage :public BasePage {
public:
	void content() {
		cout << "LOL直播" << endl;
	}
};

void test01() {
	cout << "游戏页面内容如下：" << endl;

	GamePage game;
	game.content();
	game.footer();
	game.header();
	game.leftList();

	cout << "-----------------------" << endl;
	cout << "新闻页面内容如下：" << endl;

	NewsPage news;
	news.content();
	news.footer();
	news.header();
	news.leftList();
}

int main() {
	test01();
	system("pause");
	return EXIT_SUCCESS;
}