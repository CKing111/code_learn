#include <stdio.h>
#include <stdlib.h>
#define  MAX_VERTEX_NUM 20 //最大顶点个数
#define  VertexType int //顶点数据的类型
typedef enum { false, true } bool;

//建立全局变量，存储各个顶点（事件）的最早发生时间
VertexType ve[MAX_VERTEX_NUM] = { 0 };
//建立全局变量，保存各个顶点（事件）的最晚发生时间
VertexType vl[MAX_VERTEX_NUM] = { 0 };

typedef struct ArcNode {
    int adjvex; //邻接点在数组中的位置下标
    struct ArcNode* nextarc; //指向下一个邻接点的指针
    VertexType dut;
}ArcNode;

typedef struct VNode {
    VertexType data; //顶点的数据域
    ArcNode* firstarc; //指向邻接点的指针
}VNode, AdjList[MAX_VERTEX_NUM]; //存储各链表头结点的数组

typedef struct {
    AdjList vertices; //图中顶点及各邻接点数组
    int vexnum, arcnum; //记录图中顶点数和弧数
}ALGraph;

//找到顶点在邻接表数组中的位置下标
int LocateVex(ALGraph G, VertexType u) {
    int i;
    for (i = 0; i < G.vexnum; i++) {
        if (G.vertices[i].data == u) {
            return i;
        }
    }
    return -1;
}

//邻接表存储 AOE 网（有向无环网）
void CreateAOE(ALGraph* G) {
    int i, locate;
    VertexType initial, end, dut;
    ArcNode* p = NULL;
    scanf("%d,%d", &(G->vexnum), &(G->arcnum));
    for (i = 0; i < G->vexnum; i++) {
        scanf("%d", &(G->vertices[i].data));
        G->vertices[i].firstarc = NULL;
    }
    for (i = 0; i < G->arcnum; i++) {
        scanf("%d,%d,%d", &initial, &end, &dut);
        p = (ArcNode*)malloc(sizeof(ArcNode));
        p->adjvex = LocateVex(*G, end);
        p->nextarc = NULL;
        p->dut = dut;
        locate = LocateVex(*G, initial);
        p->nextarc = G->vertices[locate].firstarc;
        G->vertices[locate].firstarc = p;
    }
}

//结构体定义栈结构
typedef struct stack {
    VertexType data;
    struct stack* next;
}stack;

//初始化栈结构
void initStack(stack** S) {
    (*S) = (stack*)malloc(sizeof(stack));
    (*S)->next = NULL;
}

//判断栈是否为空
bool StackEmpty(stack S) {
    if (S.next == NULL) {
        return true;
    }
    return false;
}

//进栈，以头插法将新结点插入到链表中
void push(stack* S, VertexType u) {
    stack* p = (stack*)malloc(sizeof(stack));
    p->data = u;
    p->next = NULL;
    p->next = S->next;
    S->next = p;
}

//弹栈函数，删除链表首元结点的同时，释放该空间，并将该结点中的数据域通过地址传值给变量i;
void pop(stack* S, VertexType* i) {
    stack* p = S->next;
    *i = p->data;
    S->next = S->next->next;
    free(p);
}

void deleStack(stack* S) {
    stack* del = NULL;
    if (S->next) {
        del = S->next;
        S->next = del->next;
        free(del);
    }
    free(S);
}

//创建一个全局指针，后续会指向一个链栈
stack* T = NULL;

//统计各顶点的入度
void FindInDegree(ALGraph G, int indegree[]) {
    int i;
    //遍历邻接表，根据各链表中数据域存储的各顶点位置下标，在indegree数组相应位置+1
    for (i = 0; i < G.vexnum; i++) {
        ArcNode* p = G.vertices[i].firstarc;
        while (p) {
            indegree[p->adjvex]++;
            p = p->nextarc;
        }
    }
}

bool TopologicalOrder(ALGraph G) {
    int index, i, indegree[MAX_VERTEX_NUM] = { 0 };//创建记录各顶点入度的数组
    //建立链栈
    stack* S = NULL;
    int count = 0;
    ArcNode* p = NULL;
    FindInDegree(G, indegree);//统计各顶点的入度
    //初始化栈
    initStack(&S);
    //查找度为0的顶点，作为起始点
    for (i = 0; i < G.vexnum; i++) {
        if (!indegree[i]) {
            push(S, i);
        }
    }
    //栈为空为结束标志
    while (!StackEmpty(*S)) {
        //弹栈，并记录栈中保存的顶点所在邻接表数组中的位置
        pop(S, &index);
        //压栈，为求各边的最晚开始时间做准备（实现拓扑排序的逆过程）
        push(T, index);
        ++count;
        //依次查找跟该顶点相关联的顶点，将它们的入度减1，然后将入度为 0 的顶点入栈
        for (p = G.vertices[index].firstarc; p; p = p->nextarc) {
            VertexType k = p->adjvex;
            if (!(--indegree[k])) {
                //顶点入度为0，入栈
                push(S, k);
            }
            //如果 index 的最早发生时间加上边的权值比 k 的最早发生时间还长，就覆盖ve数组中对应位置的值，最终结束时，ve数组中存储的就是各顶点的最早发生时间。
            if (ve[index] + p->dut > ve[k]) {
                ve[k] = ve[index] + p->dut;
            }
        }
    }
    deleStack(S);
    //如果count值小于顶点数量，表明有向图有环
    if (count < G.vexnum) {
        printf("该图有回路");
        return false;
    }
    return true;
}

//求各顶点的最晚发生时间并计算出各边的最早和最晚开始时间
void CriticalPath(ALGraph G) {
    int i, j, k;
    ArcNode* p = NULL;
    if (!TopologicalOrder(G)) {
        return;
    }
    for (i = 0; i < G.vexnum; i++) {
        vl[i] = ve[G.vexnum - 1];
    }
    //计算各个顶点的最晚发生时间
    while (!StackEmpty(*T)) {
        pop(T, &j);
        for (p = G.vertices[j].firstarc; p; p = p->nextarc) {
            k = p->adjvex;
            //构建Vl数组，在初始化时，Vl数组中每个单元都是18，如果每个边的汇点-边的权值比源点值小，就保存更小的。
            if (vl[k] - p->dut < vl[j]) {
                vl[j] = vl[k] - p->dut;
            }
        }
    }
    printf("关键路径为\n");
    //结合 ve 和 vl 数组中存储的数据，可以计算出每条边的最早开始时间和最晚开始时间，进而判断各个边是否为关键路径
    for (j = 0; j < G.vexnum; j++) {
        int ee, el;
        for (p = G.vertices[j].firstarc; p; p = p->nextarc) {
            k = p->adjvex;
            //求各边的最早开始时间e[i],等于ve数组中相应源点存储的值
            ee = ve[j];
            //求各边的最晚开始时间l[i]，等于汇点在vl数组中存储的值减改边的权值
            el = vl[k] - p->dut;
            //判断e[i]和l[i]是否相等，如果相等，该边就是关键活动，相应的用*标记；反之，边后边没标记
            if (ee == el) {
                printf("<V%d,V%d> ", j + 1, k + 1);
            }
        }
    }
}

int main() {
    ALGraph G;
    CreateAOE(&G);//创建AOE网
    initStack(&T);
    TopologicalOrder(G);
    CriticalPath(G);
    deleStack(T);
    system("pause");
    return  0;
}
