#include <stdio.h>
#include <stdlib.h>
#define MAX_VERtEX_NUM 20                   //������������
#define VRType int                          //��ʾ����֮���ϵ����������
#define VertexType int                      //ͼ�ж������������
#define States int
typedef enum { false, true }bool;           //����bool�ͳ���
bool visited[MAX_VERtEX_NUM];              //����ȫ�����飬��¼��Ƕ����Ƿ񱻷��ʹ�

typedef struct {
    VRType adj;                             //������Ȩͼ���� 1 �� 0 ��ʾ�Ƿ����ڣ����ڴ�Ȩͼ��ֱ��ΪȨֵ��
}ArcCell, AdjMatrix[MAX_VERtEX_NUM][MAX_VERtEX_NUM];

typedef struct {
    VertexType vexs[MAX_VERtEX_NUM];         //�洢ͼ�ж�������
    AdjMatrix arcs;                          //��ά���飬��¼����֮��Ĺ�ϵ
    int vexnum, arcnum;                      //��¼ͼ�Ķ������ͻ����ߣ���
}MGraph;

//�����ֵܱ�ʾ���Ķ������ṹ
typedef struct CSNode {
    VertexType data;
    struct CSNode* lchild;//���ӽ��
    struct CSNode* nextsibling;//�ֵܽ��
}*CSTree, CSNode;

typedef struct Queue {
    CSTree data;//�����д�ŵ�Ϊ�����
    struct Queue* next;
}Queue;

//���ݶ������ݣ����ض����ڶ�ά�����е�λ���±�
int LocateVex(MGraph* G, VertexType v) {
    int i = 0;
    //����һά���飬�ҵ�����v
    for (; i < G->vexnum; i++) {
        if (G->vexs[i] == v) {
            break;
        }
    }
    //����Ҳ����������ʾ��䣬����-1
    if (i > G->vexnum) {
        printf("no such vertex.\n");
        return -1;
    }
    return i;
}

//��������ͼ
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
        G->arcs[m][n].adj = 1;//����ͼ�Ķ��׾��������Խ��߶Գ�
    }
    return 1;
}

int FirstAdjVex(MGraph G, int v)
{
    int i;
    //���������±� v ���Ķ��㣬�ҵ���һ���������ڵĶ��㣬�����ظö���������±�
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
    //���������±� v ���Ķ��㣬�� w λ�ÿ�ʼ�������Һ������ڵĶ��㣬�����ظö���������±�
    for (i = w + 1; i < G.vexnum; i++) {
        if (G.arcs[v][i].adj) {
            return i;
        }
    }
    return -1;
}

//��ʼ������
void InitQueue(Queue** Q) {
    (*Q) = (Queue*)malloc(sizeof(Queue));
    (*Q)->next = NULL;
}

//���T������
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

//��ͷԪ�س�����
void DeQueue(Queue** Q, CSTree* u) {
    Queue* del = (*Q)->next;
    (*u) = (*Q)->next->data;
    (*Q)->next = (*Q)->next->next;
    free(del);
}

//�ж϶����Ƿ�Ϊ��
bool QueueEmpty(Queue* Q) {
    if (Q->next == NULL) {
        return true;
    }
    return false;
}

//�ͷŶ���ռ�õĶѿռ�
void DelQueue(Queue* Q) {
    Queue* del = NULL;
    while (Q->next) {
        del = Q->next;
        Q->next = Q->next->next;
        free(del);
    }
    free(Q);
}

//�����������������
void BFSTree(MGraph G, CSTree* T) {
    int v, w;
    CSTree q = NULL, p = NULL;
    Queue* Q = NULL;
    InitQueue(&Q);
    //��������
    EnQueue(&Q, (*T));
    //������Ϊ��ʱ��֤���������
    while (!QueueEmpty(Q)) {
        bool first = true;
        //�����׸�������
        DeQueue(&Q, &q);
        //�жϽ���е������������еľ���λ��
        v = LocateVex(&G, q->data);
        //�Ѿ����ʹ��ĸ������־λ
        visited[v] = true;
        //������ v ���������ȫ����ӣ����Ժ����ֵܱ�ʾ������������
        for (w = FirstAdjVex(G, v); w >= 0; w = NextAdjVex(G, v, w)) {
            //��־λΪfalse��֤��δ������
            if (!visited[w]) {
                //�½�һ����� p����ŵ�ǰ�����Ķ���
                p = (CSTree)malloc(sizeof(CSNode));
                p->data = G.vexs[w];
                p->lchild = NULL;
                p->nextsibling = NULL;
                //��ǰ������
                EnQueue(&Q, p);
                //���ı�־λ
                visited[w] = true;
                //���ӵĵ�һ�����㣬����Ϊ q ������
                if (first) {
                    q->lchild = p;
                    first = false;
                }
                //����������Ϊ�ֵܽ��
                else {
                    q->nextsibling = p;
                }
                q = p;
            }
        }
    }
    DelQueue(Q);
}

//���������������ɭ�ֲ�ת��Ϊ������
void BFSForest(MGraph G, CSTree* T) {
    int v;
    CSTree q = NULL, p = NULL;
    //ÿ������ı��Ϊ��ʼ��Ϊfalse
    for (v = 0; v < G.vexnum; v++) {
        visited[v] = false;
    }
    //����ͼ�����еĶ���
    for (v = 0; v < G.vexnum; v++) {
        //����ö���ı��λΪfalse��֤��δ���ʹ�
        if (!(visited[v])) {
            //�½�һ����㣬��ʾ�ö���
            p = (CSTree)malloc(sizeof(CSNode));
            p->data = G.vexs[v];
            p->lchild = NULL;
            p->nextsibling = NULL;
            //�����δ�գ���ö�����Ϊ��������
            if (!(*T)) {
                (*T) = p;
            }
            //�ö�����Ϊ�������ֵܽ��
            else {
                q->nextsibling = p;
            }
            //ÿ�ζ�Ҫ��qָ��ָ���µĽ�㣬Ϊ�´���ӽ�����̵�
            q = p;
            //�Ըý��Ϊ��ʼ�㣬�����������������
            BFSTree(G,&p);
        }
    }
}

//ǰ�����������
void PreOrderTraverse(CSTree T) {
    if (T) {
        printf("%d ", T->data);
        PreOrderTraverse(T->lchild);
        PreOrderTraverse(T->nextsibling);
    }
    return;
}

//����������������ͷ���ռ�õ��ڴ�
void DestroyBiTree(CSTree T) {
    if (T) {
        DestroyBiTree(T->lchild);//��������
        DestroyBiTree(T->nextsibling);//�����Һ���
        free(T);//�ͷŽ��ռ�õ��ڴ�
    }
}

int main() {
    MGraph G;//����һ��ͼ�ı���
    CSTree T = NULL;
    CreateDN(&G);//��ʼ��ͼ
    BFSForest(G, &T);
    PreOrderTraverse(T);
    DestroyBiTree(T);
    system("pause");
    return 0;
}
