#include<iostream>

using namespace std;

// ��ͳ��������ִ����ظ�����
// 
//// ��Ϸҳ��
//class GamePage {
//public:
//	void header() {
//		cout << "����ͷ��" << endl;
//	}
//	void footer() {
//		cout<<"�����ײ�" << endl;
//	}
//	void leftList() {
//		cout << "����������б�" << endl;
//	}
//	void content() {
//		cout << "LOL" << endl;
//	}
//};
//
//// ����ҳ��
//class NewPage {
//public:
//	void header() {
//		cout << "����ͷ��" << endl;
//	}
//	void footer() {
//		cout << "�����ײ�" << endl;
//	}
//	void leftList() {
//		cout << "����������б�" << endl;
//	}
//	void content() {
//		cout << "NEWs" << endl;
//	}
//};

// ʹ�ü̳У������ظ�����ĳ���
// BasePage:���ࡢ����
// GamePage\NewsPage: �����ࡢ���� 
// �̳з�ʽ��
// �����̳У�class ���� ��public ����{}--------���ɷ��ʸ���˽�У���������
// �����̳У�class ���� ��protected ����{}-----���ɷ��ʸ���˽�У������䱣��
// ˽�м̳У�class ���� ��private ����{}-------���ɷ��ʸ���˽�У�������˽��
class BasePage {
public:
	void header() {
		cout << "����ͷ��" << endl;
	}
	void footer() {
		cout << "�����ײ�" << endl;
	}
	void leftList() {
		cout << "����������б�" << endl;
	}
};
// ������̳�
class NewsPage :public BasePage {
public:
	void content() {
		cout << "���ᱨ��" << endl;
	}
};
// ��Ϸ��̳�
class GamePage :public BasePage {
public:
	void content() {
		cout << "LOLֱ��" << endl;
	}
};

void test01() {
	cout << "��Ϸҳ���������£�" << endl;

	GamePage game;
	game.content();
	game.footer();
	game.header();
	game.leftList();

	cout << "-----------------------" << endl;
	cout << "����ҳ���������£�" << endl;

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