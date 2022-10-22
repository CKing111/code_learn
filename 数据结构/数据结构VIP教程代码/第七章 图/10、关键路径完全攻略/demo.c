#include <stdio.h>
#include <stdlib.h>
#define  MAX_VERTEX_NUM 20 //��󶥵����
#define  VertexType int //�������ݵ�����
typedef enum { false, true } bool;

//����ȫ�ֱ������洢�������㣨�¼��������緢��ʱ��
VertexType ve[MAX_VERTEX_NUM] = { 0 };
//����ȫ�ֱ���������������㣨�¼�����������ʱ��
VertexType vl[MAX_VERTEX_NUM] = { 0 };

typedef struct ArcNode {
    int adjvex; //�ڽӵ��������е�λ���±�
    struct ArcNode* nextarc; //ָ����һ���ڽӵ��ָ��
    VertexType dut;
}ArcNode;

typedef struct VNode {
    VertexType data; //�����������
    ArcNode* firstarc; //ָ���ڽӵ��ָ��
}VNode, AdjList[MAX_VERTEX_NUM]; //�洢������ͷ��������

typedef struct {
    AdjList vertices; //ͼ�ж��㼰���ڽӵ�����
    int vexnum, arcnum; //��¼ͼ�ж������ͻ���
}ALGraph;

//�ҵ��������ڽӱ������е�λ���±�
int LocateVex(ALGraph G, VertexType u) {
    int i;
    for (i = 0; i < G.vexnum; i++) {
        if (G.vertices[i].data == u) {
            return i;
        }
    }
    return -1;
}

//�ڽӱ�洢 AOE ���������޻�����
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

//�ṹ�嶨��ջ�ṹ
typedef struct stack {
    VertexType data;
    struct stack* next;
}stack;

//��ʼ��ջ�ṹ
void initStack(stack** S) {
    (*S) = (stack*)malloc(sizeof(stack));
    (*S)->next = NULL;
}

//�ж�ջ�Ƿ�Ϊ��
bool StackEmpty(stack S) {
    if (S.next == NULL) {
        return true;
    }
    return false;
}

//��ջ����ͷ�巨���½����뵽������
void push(stack* S, VertexType u) {
    stack* p = (stack*)malloc(sizeof(stack));
    p->data = u;
    p->next = NULL;
    p->next = S->next;
    S->next = p;
}

//��ջ������ɾ��������Ԫ����ͬʱ���ͷŸÿռ䣬�����ý���е�������ͨ����ַ��ֵ������i;
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

//����һ��ȫ��ָ�룬������ָ��һ����ջ
stack* T = NULL;

//ͳ�Ƹ���������
void FindInDegree(ALGraph G, int indegree[]) {
    int i;
    //�����ڽӱ����ݸ�������������洢�ĸ�����λ���±꣬��indegree������Ӧλ��+1
    for (i = 0; i < G.vexnum; i++) {
        ArcNode* p = G.vertices[i].firstarc;
        while (p) {
            indegree[p->adjvex]++;
            p = p->nextarc;
        }
    }
}

bool TopologicalOrder(ALGraph G) {
    int index, i, indegree[MAX_VERTEX_NUM] = { 0 };//������¼��������ȵ�����
    //������ջ
    stack* S = NULL;
    int count = 0;
    ArcNode* p = NULL;
    FindInDegree(G, indegree);//ͳ�Ƹ���������
    //��ʼ��ջ
    initStack(&S);
    //���Ҷ�Ϊ0�Ķ��㣬��Ϊ��ʼ��
    for (i = 0; i < G.vexnum; i++) {
        if (!indegree[i]) {
            push(S, i);
        }
    }
    //ջΪ��Ϊ������־
    while (!StackEmpty(*S)) {
        //��ջ������¼ջ�б���Ķ��������ڽӱ������е�λ��
        pop(S, &index);
        //ѹջ��Ϊ����ߵ�����ʼʱ����׼����ʵ���������������̣�
        push(T, index);
        ++count;
        //���β��Ҹ��ö���������Ķ��㣬�����ǵ���ȼ�1��Ȼ�����Ϊ 0 �Ķ�����ջ
        for (p = G.vertices[index].firstarc; p; p = p->nextarc) {
            VertexType k = p->adjvex;
            if (!(--indegree[k])) {
                //�������Ϊ0����ջ
                push(S, k);
            }
            //��� index �����緢��ʱ����ϱߵ�Ȩֵ�� k �����緢��ʱ�仹�����͸���ve�����ж�Ӧλ�õ�ֵ�����ս���ʱ��ve�����д洢�ľ��Ǹ���������緢��ʱ�䡣
            if (ve[index] + p->dut > ve[k]) {
                ve[k] = ve[index] + p->dut;
            }
        }
    }
    deleStack(S);
    //���countֵС�ڶ�����������������ͼ�л�
    if (count < G.vexnum) {
        printf("��ͼ�л�·");
        return false;
    }
    return true;
}

//��������������ʱ�䲢��������ߵ����������ʼʱ��
void CriticalPath(ALGraph G) {
    int i, j, k;
    ArcNode* p = NULL;
    if (!TopologicalOrder(G)) {
        return;
    }
    for (i = 0; i < G.vexnum; i++) {
        vl[i] = ve[G.vexnum - 1];
    }
    //������������������ʱ��
    while (!StackEmpty(*T)) {
        pop(T, &j);
        for (p = G.vertices[j].firstarc; p; p = p->nextarc) {
            k = p->adjvex;
            //����Vl���飬�ڳ�ʼ��ʱ��Vl������ÿ����Ԫ����18�����ÿ���ߵĻ��-�ߵ�Ȩֵ��Դ��ֵС���ͱ����С�ġ�
            if (vl[k] - p->dut < vl[j]) {
                vl[j] = vl[k] - p->dut;
            }
        }
    }
    printf("�ؼ�·��Ϊ\n");
    //��� ve �� vl �����д洢�����ݣ����Լ����ÿ���ߵ����翪ʼʱ�������ʼʱ�䣬�����жϸ������Ƿ�Ϊ�ؼ�·��
    for (j = 0; j < G.vexnum; j++) {
        int ee, el;
        for (p = G.vertices[j].firstarc; p; p = p->nextarc) {
            k = p->adjvex;
            //����ߵ����翪ʼʱ��e[i],����ve��������ӦԴ��洢��ֵ
            ee = ve[j];
            //����ߵ�����ʼʱ��l[i]�����ڻ����vl�����д洢��ֵ���ıߵ�Ȩֵ
            el = vl[k] - p->dut;
            //�ж�e[i]��l[i]�Ƿ���ȣ������ȣ��ñ߾��ǹؼ������Ӧ����*��ǣ���֮���ߺ��û���
            if (ee == el) {
                printf("<V%d,V%d> ", j + 1, k + 1);
            }
        }
    }
}

int main() {
    ALGraph G;
    CreateAOE(&G);//����AOE��
    initStack(&T);
    TopologicalOrder(G);
    CriticalPath(G);
    deleStack(T);
    system("pause");
    return  0;
}
