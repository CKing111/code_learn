#include <stdio.h>
#include <stdlib.h>
#define MAX_VERtEX_NUM 20                   //顶点的最大数量
#define VRType int                          //表示顶点之间关系的数据类型
#define VertexType int                      //图中顶点的数据类型
#define States int
typedef enum { false, true }bool;           //定义bool型常量
bool visited[MAX_VERtEX_NUM];              //设置全局数组，记录标记顶点是否被访问过

typedef struct {
    VRType adj;                             //对于无权图，用 1 或 0 表示是否相邻；对于带权图，直接为权值。
}ArcCell, AdjMatrix[MAX_VERtEX_NUM][MAX_VERtEX_NUM];

typedef struct {
    VertexType vexs[MAX_VERtEX_NUM];         //存储图中顶点数据
    AdjMatrix arcs;                          //二维数组，记录顶点之间的关系
    int vexnum, arcnum;                      //记录图的顶点数和弧（边）数
}MGraph;

//孩子兄弟表示法的二叉树结构
typedef struct CSNode {
    VertexType data;
    struct CSNode* lchild;//孩子结点
    struct CSNode* nextsibling;//兄弟结点
}*CSTree, CSNode;

//根据顶点数据，返回顶点在二维数组中的位置下标
int LocateVex(MGraph* G, VertexType v) {
    int i = 0;
    //遍历一维数组，找到变量v
    for (; i < G->vexnum; i++) {
        if (G->vexs[i] == v) {
            break;
        }
    }
    //如果找不到，输出提示语句，返回-1
    if (i > G->vexnum) {
        printf("no such vertex.\n");
        return -1;
    }
    return i;
}

//构造无向图
States CreateDN(MGraph* G) {
    int i, j, n, m;
    int v1, v2;
    scanf("%d,%d", &(G->vexnum), &(G->arcnum));
    for (i = 0; i < G->vexnum; i++) {
        scanf("%d", &(G->vexs[i]));
    }
    for (i = 0; i < G->vexnum; i++) {
        for (j = 0; j < G->vexnum; j++) {
            G->arcs[i][j].adj = 0;
        }
    }
    for (i = 0; i < G->arcnum; i++) {
        scanf("%d,%d", &v1, &v2);
        n = LocateVex(G, v1);
        m = LocateVex(G, v2);
        if (m == -1 || n == -1) {
            printf("no this vertex\n");
            return -1;
        }
        G->arcs[n][m].adj = 1;
        G->arcs[m][n].adj = 1;//无向图的二阶矩阵沿主对角线对称
    }
    return 1;
}

int FirstAdjVex(MGraph G, int v)
{
    int i;
    //对于数组下标 v 处的顶点，找到第一个和它相邻的顶点，并返回该顶点的数组下标
    for (i = 0; i < G.vexnum; i++) {
        if (G.arcs[v][i].adj) {
            return i;
        }
    }
    return -1;
}

int NextAdjVex(MGraph G, int v, int w)
{
    int i;
    //对于数组下标 v 处的顶点，从 w 位置开始继续查找和它相邻的顶点，并返回该顶点的数组下标
    for (i = w + 1; i < G.vexnum; i++) {
        if (G.arcs[v][i].adj) {
            return i;
        }
    }
    return -1;
}

void CreadDFSTree(MGraph G, int v, CSTree* T) {
    int i, w;
    CSTree p = NULL, q = NULL;
    bool first = true;
    //将正在访问的该顶点的标志位设为true
    visited[v] = true;
    //依次遍历所有紧邻 v 的其它顶点
    for (w = FirstAdjVex(G, v); w >= 0; w = NextAdjVex(G, v, w)) {
        //如果该临界点标志位为false，说明还未访问
        if (!visited[w]) {
            //用一个结点存储当前顶点
            p = (CSTree)malloc(sizeof(CSNode));
            p->data = G.vexs[w];
            p->lchild = NULL;
            p->nextsibling = NULL;
            //找到的第一个紧邻 v 的顶点作为它的左孩子结点，后续找到的顶点作为它左孩子的兄弟结点
            if (first) {
                (*T)->lchild = p;
                first = false;
            }
            //否则，为兄弟结点
            else {
                q->nextsibling = p;
            }
            q = p;
            //从当前访问的顶点出发，继续访问紧邻它且尚未访问的顶点
            CreadDFSTree(G, w, &q);
        }
    }
}

/*获得指定连通图 G 对应的深度优先生成树（二叉树）
* G：要遍历的连通图
* v: 从哪个顶点开始遍历，v 是该顶点在顶点数组中的位置下标
* T: 存储生成树
*/
void DFSTree(MGraph G, CSTree* T) {
    int v;
    CSTree p = NULL;
    //每个顶点的标记为初始化为false
    for (v = 0; v < G.vexnum; v++) {
        visited[v] = false;
    }
    //任选一个顶点，作为生成树的根结点
    p = (CSTree)malloc(sizeof(CSNode));
    p->data = G.vexs[0];
    p->lchild = NULL;
    p->nextsibling = NULL;
    (*T) = p;
    //从 vexs[0] 处顶点出发，构建生成树
    CreadDFSTree(G, 0, &p);
}

//先序遍历二叉树
void PreOrderTraverse(CSTree T) {
    if (T) {
        printf("%d ", T->data);
        PreOrderTraverse(T->lchild);
        PreOrderTraverse(T->nextsibling);
    }
    return;
}

//后序遍历二叉树，释放树占用的内存
void DestroyBiTree(CSTree T) {
    if (T) {
        DestroyBiTree(T->lchild);//销毁左孩子
        DestroyBiTree(T->nextsibling);//销毁右孩子
        free(T);//释放结点占用的内存
    }
}

int main() {
    MGraph G;//建立一个图的变量
    CSTree T = NULL;
    CreateDN(&G);//初始化图
    DFSTree(G, &T);
    //先序遍历二叉树
    PreOrderTraverse(T);
    DestroyBiTree(T);
    return 0;
}
