# include <iostream>
# include <string>
# include <fstream>
# include <stack>
using namespace std;

const int max_size = 56;

struct element
{
    char data;// 保存的符号
    int weight;// 权值
    int lchild,rchild,parent;// 左孩子，右孩子，根结点
};

class HuffmanTree {
private:
    // 这个是从文本中统计出来的出现的字符的数量
    int num;
    // 保存各个叶子的权值
    int w[max_size];
    // 
    element huffTree[2*max_size-1];
    // 判断位置
    int indexOf(char ch);
    // 选择最小值
    void Select(int &i1, int &i2);
    // 读取文件
    string read_file();
    // 保存压缩文件
    void save_code(int code_cache[], int code_cache_size);
    // 统计出现的字母和频率
    void CountLetter(string& str);
public:
    // 构造函数
    HuffmanTree() ;
    // 打印输出
    void print();
    // 转码压缩
    void condense(string str);
    // 解码
    void decoding();
};

/**
  * 哈夫曼算法-解码
  */
void HuffmanTree::decoding() {
    string str;
    ifstream in("456.txt",ios::in);
    in>>str;
    in.close();
    ofstream out("789.txt",ios::out);
    int root = 2*num-2;
    int temp = root;
    for (int i = 0;i < str.size(); i++) {
        if (str.at(i) == '0') {
            if (huffTree[temp].lchild != -1) {
                temp = huffTree[temp].lchild;
                if (huffTree[temp].lchild == -1) {
                    out<<huffTree[temp].data;
                    temp = root;
                }
            } 
        } else {// is 0
            if (huffTree[temp].rchild != -1) {
                temp = huffTree[temp].rchild;
                if (huffTree[temp].rchild == -1) {
                    out<<huffTree[temp].data;
                    temp = root;
                }
            } 
        }
    }
    out.close();
}

/**
  * 哈夫曼算法-计算频率
  */
void HuffmanTree::CountLetter(string& str) {
    int size = 52;
    char ch[size];
    int weight[size];
    num = 0;
    ch[0] = 'A';
    ch[26] = 'a';
    for (int i = 1;i < size / 2; i++) {
        ch[i] = ch[i-1] + 1;
        ch[i+26] = ch[i+26-1] + 1;
    }
    for (int i = 0;i < size; i++) {
        weight[i] = 0;
    }
    //char c=get(); int ch[c-'a']++;
    for (int i = 0;i < str.size(); i++) {
        if (str.at(i) > 64 && str.at(i) < 91) {
            weight[str.at(i) - 'A']++;
        } 
        if (str.at(i) > 96 && str.at(i) < 123) {
            weight[26 + str.at(i) - 'a']++;
        } 
    }
    for (int i = 0;i < size; i++) {
        if (weight[i] != 0) {
            huffTree[num].data = ch[i];
            huffTree[num].weight = weight[i];
            w[num] = weight[i];
            num++;
        }
    }
}

/**
  * 哈夫曼算法-读取文件
  * 这里没有处理当数据很庞大的情况
  */
string HuffmanTree::read_file() {
    string str;
    ifstream in("123.txt",ios::in);
    in>>str;
    in.close();
    return str;
}

/**
  * 哈夫曼算法-压缩
  * @param code_cache[] 转码数据
  * @param code_cache_size 转码数据数组的长度
  */
void HuffmanTree::save_code(int code_cache[], int code_cache_size) {
    ofstream out("456.txt",ios::out);
    // for (int i = code_cache_size - 1;i >= 0; i--) {
    //  out<<code_cache[i];
    // }
    for (int i = 0;i < code_cache_size; i++) {
        out<<code_cache[i];
    }
    out.close();
}

/**
  * 哈夫曼算法-转码压缩
  * @param huffTree[] 这个是哈夫曼数组
  * @param str 是文件流中的字符串
  */
void HuffmanTree::condense(string str) {
    // 缓存转码数据数组的最大缓存大小
    int code_cache_max_size = 256;
    // temp 
    stack<int> s;
    // 缓存转码数据数组
    int code_cache[code_cache_max_size];
    // 缓存游标
    int code_cache_index = 0;
    // 判断是左子树还是右子树
    int Judge_Direction;
    // 是否上溢    
    bool isOverflow = false;                                         
    for (int i = 0; i < str.size(); i++) {
        int index = indexOf(str.at(i));
        Judge_Direction = huffTree[index].parent;
        while (Judge_Direction != -1) { 
            if (index == huffTree[Judge_Direction].lchild) {
                s.push(0);
            } else {
                s.push(1);
            }
            index = Judge_Direction;
            Judge_Direction = huffTree[Judge_Direction].parent;
        }
        while (!s.empty()) {
            code_cache[code_cache_index++] = s.top();
            s.pop();
        }
        if (code_cache_index % code_cache_max_size == code_cache_max_size - 1) {
            save_code(code_cache, code_cache_max_size);
            code_cache_index = 0;
            isOverflow = true;
        }
    }
    if (!isOverflow) {
        save_code(code_cache, code_cache_index);
    }
    // for (int i =0; i < code_cache_index;i++) {
    //  cout<<code_cache[i];
    // }
    // cout<<endl;
}

/**
  * 哈夫曼算法-查找位置
  * @param huffTree[] 这个是哈夫曼数组
  * @param ch 想要查找的位置
  */
int HuffmanTree::indexOf(char ch) {
    for (int i = 0;i < num; i++) {
        if (huffTree[i].data == ch) {
            return i;
        }
    }
    return -1;
}

/**
  * 哈夫曼算法-输出
  * @param huffTree[] 这个是哈夫曼数组
  */
void HuffmanTree::print() {
    for (int i = 0;i < 2*num-1; i++) {
        cout<<huffTree[i].weight<<"\t"<<huffTree[i].parent<<"\t"<<huffTree[i].lchild<<"\t"<<huffTree[i].rchild<<endl;
    }
}

/**
  * 哈夫曼算法-选择出最小的两个下标
  * @param huffTree[] 这个是哈夫曼数组
  * @param i1 是第一个下标
  * @param i2 是第二个下标
  */
void HuffmanTree::Select(int &i1, int &i2) {
    int i;
    bool button = true;
    for (i = 0;i < 2*num-1; i++) {
        if (button) {
            if (huffTree[i].parent == -1) {
                i1 = i;
                button = false;
            }
        } else {
            if (huffTree[i].parent == -1) {
                i2 = i;
                break;
            }
        }

    }
    for (i = 0;i < 2*num-1; i++) {
        if (i2 != i &&  huffTree[i].parent == -1 && huffTree[i].weight > 0) {
            if (huffTree[i1].weight > huffTree[i].weight) {
                i1 = i;
            }
        }
    }
    for (i = 0;i < 2*num-1; i++) {
        if (i1 != i && huffTree[i].parent == -1 && huffTree[i].weight > 0) {
            if (huffTree[i2].weight > huffTree[i].weight) {
                i2 = i;
            }
        }
    }
}

/**
  * 哈夫曼算法
  * @param huffTree[] 这个是哈夫曼数组
  * @param w[] 权值数组
  * @param n 字符的数量
  */
HuffmanTree::HuffmanTree() {
    string str = read_file();
    CountLetter(str);
    for (int i = 0;i < 2*num-1;i++) {
        huffTree[i].parent = -1;
        huffTree[i].lchild = -1;
        huffTree[i].rchild = -1;
    }
    for (int i = 0;i < 2*num - 1; i++) {
        if (i < num) {
            huffTree[i].weight = w[i];// 构造2*n-1棵只含有根节点的二叉树
        } else {
            huffTree[i].weight = 0;
        }
    }
    for (int k = num;k < 2*num-1; k++) {
        int i1,i2;
        // 查找全职最小的两个根节点，下标为i1和i2
        Select(i1, i2); 
        //cout<<"合并结点：i1 = "<<i1<<",i2 = "<<i2<<endl;
        // 将i1和i2合并。且i1和i2的双亲都是k
        huffTree[i1].parent = k;
        huffTree[i2].parent = k;
        huffTree[k].weight = huffTree[i1].weight + huffTree[i2].weight;
        huffTree[k].lchild = i1;
        huffTree[k].rchild = i2;
    }
    cout<<"------------------------"<<endl;
    print();
    cout<<"------------------------"<<endl;
    condense(str);
    decoding();
}

int main() {
    new HuffmanTree();
    return 0;
}