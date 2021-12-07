#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
using namespace std;
struct Node {
	Node(double d, Node* l = NULL, Node* r = NULL, Node* f = NULL) :data(d), left(l), right(r), father(f) {}
	double data;
	Node* father, * left, * right;	//父，左右孩子
	string code;	//存储编码
};
typedef Node* Tree; //通过中序，构建编码
void creatCode(Node* node, string s) { 
	if (node != NULL) {
		creatCode(node->left, s + '0');
		if (node->left == NULL && node->right == NULL)		//是叶子节点就更新编码 node->code = s;
		creatCode(node->right, s + '1');
	}
};

int main() {
	vector<double> w;
	vector<Node*> node;
	double tmp;
	Tree tree;
	cout << "输入权值，数字后紧跟回车结束：";
	do {
		cin >> tmp;
		w.push_back(tmp);
	} while (getchar() != '\n');
	sort(w.begin(), w.end(), greater<double>());	//降序排序			greater()表示左大于右，降序
	for (int i = 0; i < w.size(); i++)
		node.push_back(new Node(w[i]));
	vector<Node*> out = node;
	Node * left, *right;
	do {
		right = node.back(); node.pop_back();		//取出最小的两个
		left = node.back(); node.pop_back();
		node.push_back(new Node(left->data + right->data, left, right));		//将新结点（求和）推进数组中
		left->father = node.back();			//更新父结点
		right->father = node.back();
		out.push_back(node.back());			//存储此结点
		for (int i = node.size() - 1; i > 0 && node[i]->data > node[i - 1]->data; i--);		//从末尾冒泡，排序 swap(node[i], node[i - 1]);
	} while (node.size() != 1);	  //构建树结构
	tree = node.front();			//剩余的一个结点即根结点
	creatCode(tree, ""); printf("结点\t父结点\t左孩子\t右孩子\t编码\n");
	for (int i = 0; i < out.size(); i++)
		printf("%.2lf\t%.2lf\t%.2lf\t%.2lf\t%s\n", out[i]->data, out[i]->father == NULL ? 0 : out[i]->father->data, 
					out[i]->left == NULL ? 0 : out[i]->left->data, out[i]->right == NULL ? 0 : out[i]->right->data, out[i]->code.c_str());
	return 0;
}