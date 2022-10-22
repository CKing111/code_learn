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

//���������������������ת��Ϊ������
void DFSTree(MGraph G, int v, CSTree* T) {
    int w;
    bool first = true;
    CSTree p = NULL, q = NULL;
    //�����ڷ��ʵĸö���ı�־λ��Ϊtrue
    visited[v] = true;
    //���α����ö���������ڽӵ�
    for (w = FirstAdjVex(G, v); w >= 0; w = NextAdjVex(G, v, w)) {
        //������ٽ���־λΪfalse��˵����δ����
        if (!visited[w]) {
            //Ϊ���ڽӵ��ʼ��Ϊ���
            p = (CSTree)malloc(sizeof(CSNode));
            p->data = G.vexs[w];
            p->lchild = NULL;
            p->nextsibling = NULL;
            //�ҵ��ĵ�һ������ v �Ķ�����Ϊ�������ӽ�㣬�����ҵ��Ķ�����Ϊ�����ӵ��ֵܽ��
            if (first) {
                (*T)->lchild = p;
                first = false;
            }
            //����Ϊ�ֵܽ��
            else {
                q->nextsibling = p;
            }
            q = p;
            //�ӵ�ǰ����������������ʽ������Ķ���
            DFSTree(G, w, &q);
        }
    }
}

//���������������ɭ�֣���ת��Ϊһ���ö�����
void DFSForest(MGraph G, CSTree* T) {
    int v;
    CSTree q = NULL, p = NULL;
    //ÿ������ı��λ��ʼ��Ϊfalse
    for (v = 0; v < G.vexnum; v++) {
        visited[v] = false;
    }
    //��ÿ����������������������������
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
            DFSTree(G, v, &p);
        }
    }
}

//�������������
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
    CSTree T = NULL;
    MGraph G;//����һ��ͼ�ı���
    CreateDN(&G);//��ʼ��ͼ
    DFSForest(G, &T);
    //�������������
    PreOrderTraverse(T);
    DestroyBiTree(T);
    return 0;
}
