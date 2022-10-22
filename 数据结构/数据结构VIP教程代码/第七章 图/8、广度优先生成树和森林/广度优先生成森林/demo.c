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

typedef struct Queue {
    CSTree data;//队列中存放的为树结点
    struct Queue* next;
}Queue;

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

//初始化队列
void InitQueue(Queue** Q) {
    (*Q) = (Queue*)malloc(sizeof(Queue));
    (*Q)->next = NULL;
}

//结点T进队列
void EnQueue(Queue** Q, CSTree T) {
    Queue* temp = (*Q);
    Queue* element = (Queue*)malloc(sizeof(Queue));
    element->data = T;
    element->next = NULL;
    while (temp->next != NULL) {
        temp = temp->next;
    }
    temp->next = element;
}

//队头元素出队列
void DeQueue(Queue** Q, CSTree* u) {
    Queue* del = (*Q)->next;
    (*u) = (*Q)->next->data;
    (*Q)->next = (*Q)->next->next;
    free(del);
}

//判断队列是否为空
bool QueueEmpty(Queue* Q) {
    if (Q->next == NULL) {
        return true;
    }
    return false;
}

//释放队列占用的堆空间
void DelQueue(Queue* Q) {
    Queue* del = NULL;
    while (Q->next) {
        del = Q->next;
        Q->next = Q->next->next;
        free(del);
    }
    free(Q);
}

//构建广度优先生成树
void BFSTree(MGraph G, CSTree* T) {
    int v, w;
    CSTree q = NULL, p = NULL;
    Queue* Q = NULL;
    InitQueue(&Q);
    //根结点入队
    EnQueue(&Q, (*T));
    //当队列为空时，证明遍历完成
    while (!QueueEmpty(Q)) {
        bool first = true;
        //队列首个结点出队
        DeQueue(&Q, &q);
        //判断结点中的数据在数组中的具体位置
        v = LocateVex(&G, q->data);
        //已经访问过的更改其标志位
        visited[v] = true;
        //将紧邻 v 的其它结点全部入队，并以孩子兄弟表示法构建生成树
        for (w = FirstAdjVex(G, v); w >= 0; w = NextAdjVex(G, v, w)) {
            //标志位为false，证明未遍历过
            if (!visited[w]) {
                //新建一个结点 p，存放当前遍历的顶点
                p = (CSTree)malloc(sizeof(CSNode));
                p->data = G.vexs[w];
                p->lchild = NULL;
                p->nextsibling = NULL;
                //当前结点入队
                EnQueue(&Q, p);
                //更改标志位
                visited[w] = true;
                //出队的第一个顶点，设置为 q 的左孩子
                if (first) {
                    q->lchild = p;
                    first = false;
                }
                //否则设置其为兄弟结点
                else {
                    q->nextsibling = p;
                }
                q = p;
            }
        }
    }
    DelQueue(Q);
}

//广度优先搜索生成森林并转化为二叉树
void BFSForest(MGraph G, CSTree* T) {
    int v;
    CSTree q = NULL, p = NULL;
    //每个顶点的标记为初始化为false
    for (v = 0; v < G.vexnum; v++) {
        visited[v] = false;
    }
    //遍历图中所有的顶点
    for (v = 0; v < G.vexnum; v++) {
        //如果该顶点的标记位为false，证明未访问过
        if (!(visited[v])) {
            //新建一个结点，表示该顶点
            p = (CSTree)malloc(sizeof(CSNode));
            p->data = G.vexs[v];
            p->lchild = NULL;
            p->nextsibling = NULL;
            //如果树未空，则该顶点作为树的树根
            if (!(*T)) {
                (*T) = p;
            }
            //该顶点作为树根的兄弟结点
            else {
                q->nextsibling = p;
            }
            //每次都要把q指针指向新的结点，为下次添加结点做铺垫
            q = p;
            //以该结点为起始点，构建广度优先生成树
            BFSTree(G,&p);
        }
    }
}

//前序遍历二叉树
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
    BFSForest(G, &T);
    PreOrderTraverse(T);
    DestroyBiTree(T);
    system("pause");
    return 0;
}
