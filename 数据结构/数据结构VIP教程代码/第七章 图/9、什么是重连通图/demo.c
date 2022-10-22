#include<stdio.h>
#include<stdlib.h>
#define  MAX_VERTEX_NUM 20  //最大顶点个数
#define  VertexType char    //图中顶点的类型

typedef enum { false, true }bool;

int visit[MAX_VERTEX_NUM] = { 0 };
int low[MAX_VERTEX_NUM] = { 0 };
int count = 1;

typedef struct ArcNode {
    int adjvex; // 存储弧，即另一端顶点在数组中的下标
    struct ArcNode* nextarc; // 指向下一个结点
}ArcNode;

typedef struct VNode {
    VertexType data;   //顶点的数据域
    ArcNode* firstarc; //指向下一个结点
}VNode, AdjList[MAX_VERTEX_NUM]; //存储各链表首元结点的数组

typedef struct {
    AdjList vertices;   //存储图的邻接表
    int vexnum, arcnum; //图中顶点数以及弧数
}ALGraph;

void CreateGraph(ALGraph* graph) {
    int i, j;
    char VA, VB;
    int indexA, indexB;
    ArcNode* node = NULL;
    printf("输入顶点和边的数目：\n");
    scanf("%d %d", &(graph->vexnum), &(graph->arcnum));
    //清空缓冲区
    scanf("%*[^\n]"); scanf("%*c");
    printf("输入各个顶点的值：\n");
    for (i = 0; i < graph->vexnum; i++) {
        scanf("%c", &(graph->vertices[i].data));
        getchar();
        graph->vertices[i].firstarc = NULL;
    }
    printf("输入边的信息(a b 表示边 a-b)：\n");
    //输入弧的信息，并为弧建立结点，链接到对应的链表上
    for (i = 0; i < graph->arcnum; i++) {
        scanf("%c %c", &VA, &VB);
        getchar();
        //找到 VA 和 VB 在邻接表中的位置下标
        for (j = 0; j < graph->vexnum; j++) {
            if (graph->vertices[j].data == VA) {
                indexA = j;
            }
            if (graph->vertices[j].data == VB) {
                indexB = j;
            }
        }
        //添加从 A 到 B 的弧
        node = (ArcNode*)malloc(sizeof(ArcNode));
        node->adjvex = indexB;
        node->nextarc = graph->vertices[indexA].firstarc;
        graph->vertices[indexA].firstarc = node;

        //添加从 B 到 A 的弧
        node = (ArcNode*)malloc(sizeof(ArcNode));
        node->adjvex = indexA;
        node->nextarc = graph->vertices[indexB].firstarc;
        graph->vertices[indexB].firstarc = node;
    }
}

//判断 G.vertices[v].data 是否为关结点
//函数执行结束时，会找到 v 结点（或其孩子结点）相连的祖先结点的访问次序号，如果有多个祖先结点，找到的是值最小的次序号，并存储在 low[v] 的位置
void DFSArticul(ALGraph G, int v) {
    int min, w;
    //假设当前结点不是关结点
    bool isAritcul = false;
    ArcNode* p = NULL;
    //记录当前结点被访问的次序号
    visit[v] = min = ++count;
    //遍历每个和 v 相连的结点，有些是 v 的孩子结点，有些是祖先结点
    for(p = G.vertices[v].firstarc; p ; p = p->nextarc)
    {
        w = p->adjvex;
        //如果 visit[w] 的值为 0，表示 w 是 v 的孩子结点
        if (visit[w] == 0)
        {
            //深度优先遍历，继续判断 w 是否为关结点，计算出 low[w] 的值
            DFSArticul(G, w);
            //如果 w 结点对应的 low 值比 min 值小，表明 w 与 v 的某个祖先结点建立着连接，用 min 变量记录下 low[w] 的值
            if (low[w] < min) {
                min = low[w];
            }
            // 如果 v 结点的访问次序号 <= 孩子结点记录的祖先结点的访问次序号，表明孩子结点并没有和 v 的祖宗结点建立连接，此时 v 就是关结点
            if (visit[v] <= low[w]) {
                // v 可能有多个孩子结点，visit[v] <= low[w] 条件可能成立多次。v 是关结点的信息只需要输出一次即可。
                if (isAritcul == false) {
                    isAritcul = true;
                    printf("%c ", G.vertices[v].data);
                }
            }
        }
        //如果 w 之前访问过，表明 w 是 v 的祖先结点，直接更新 min 记录的次序号
        else if(visit[w] < min)
        {
            min = visit[w];
        }
    }
    //min 变量的值经过多次更新，最终记录的是和 v 结点（或其孩子结点）相连的祖先结点的访问次序号，如果有多个祖先结点，min 记录的是值最小的次序号
    low[v] = min;
}

void FindArticul(ALGraph G) {
    //以 G.vertices[0] 为深度优先生成树的树根
    ArcNode* p = G.vertices[0].firstarc;
    visit[0] = 1;
    printf("关节点是：\n");
    //从树根顶点的第一个孩子结点出发进行深度优先搜索，查找关结点
    DFSArticul(G, p->adjvex);
    //如果从第一个孩子结点点出发，不能遍历完所有的顶点，表明树根不止有一个孩子结点，树根就是关结点
    if (count < G.vexnum) {
        printf("%c ", G.vertices[0].data);
        //继续判断别的子树，查找关结点
        while (p->nextarc) {
            p = p->nextarc;
            if (visit[p->adjvex] == 0) {
                DFSArticul(G, p->adjvex);
            }
        }
    }
}

int main() {
    ALGraph graph;
    CreateGraph(&graph);
    FindArticul(graph);
    system("pause");
    return 0;
}
