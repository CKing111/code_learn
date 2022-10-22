#include<stdio.h>
#include<stdlib.h>
#define  MAX_VERTEX_NUM 20  //��󶥵����
#define  VertexType char    //ͼ�ж��������

typedef enum { false, true }bool;

int visit[MAX_VERTEX_NUM] = { 0 };
int low[MAX_VERTEX_NUM] = { 0 };
int count = 1;

typedef struct ArcNode {
    int adjvex; // �洢��������һ�˶����������е��±�
    struct ArcNode* nextarc; // ָ����һ�����
}ArcNode;

typedef struct VNode {
    VertexType data;   //�����������
    ArcNode* firstarc; //ָ����һ�����
}VNode, AdjList[MAX_VERTEX_NUM]; //�洢��������Ԫ��������

typedef struct {
    AdjList vertices;   //�洢ͼ���ڽӱ�
    int vexnum, arcnum; //ͼ�ж������Լ�����
}ALGraph;

void CreateGraph(ALGraph* graph) {
    int i, j;
    char VA, VB;
    int indexA, indexB;
    ArcNode* node = NULL;
    printf("���붥��ͱߵ���Ŀ��\n");
    scanf("%d %d", &(graph->vexnum), &(graph->arcnum));
    //��ջ�����
    scanf("%*[^\n]"); scanf("%*c");
    printf("������������ֵ��\n");
    for (i = 0; i < graph->vexnum; i++) {
        scanf("%c", &(graph->vertices[i].data));
        getchar();
        graph->vertices[i].firstarc = NULL;
    }
    printf("����ߵ���Ϣ(a b ��ʾ�� a-b)��\n");
    //���뻡����Ϣ����Ϊ��������㣬���ӵ���Ӧ��������
    for (i = 0; i < graph->arcnum; i++) {
        scanf("%c %c", &VA, &VB);
        getchar();
        //�ҵ� VA �� VB ���ڽӱ��е�λ���±�
        for (j = 0; j < graph->vexnum; j++) {
            if (graph->vertices[j].data == VA) {
                indexA = j;
            }
            if (graph->vertices[j].data == VB) {
                indexB = j;
            }
        }
        //��Ӵ� A �� B �Ļ�
        node = (ArcNode*)malloc(sizeof(ArcNode));
        node->adjvex = indexB;
        node->nextarc = graph->vertices[indexA].firstarc;
        graph->vertices[indexA].firstarc = node;

        //��Ӵ� B �� A �Ļ�
        node = (ArcNode*)malloc(sizeof(ArcNode));
        node->adjvex = indexA;
        node->nextarc = graph->vertices[indexB].firstarc;
        graph->vertices[indexB].firstarc = node;
    }
}

//�ж� G.vertices[v].data �Ƿ�Ϊ�ؽ��
//����ִ�н���ʱ�����ҵ� v ��㣨���亢�ӽ�㣩���������Ƚ��ķ��ʴ���ţ�����ж�����Ƚ�㣬�ҵ�����ֵ��С�Ĵ���ţ����洢�� low[v] ��λ��
void DFSArticul(ALGraph G, int v) {
    int min, w;
    //���赱ǰ��㲻�ǹؽ��
    bool isAritcul = false;
    ArcNode* p = NULL;
    //��¼��ǰ��㱻���ʵĴ����
    visit[v] = min = ++count;
    //����ÿ���� v �����Ľ�㣬��Щ�� v �ĺ��ӽ�㣬��Щ�����Ƚ��
    for(p = G.vertices[v].firstarc; p ; p = p->nextarc)
    {
        w = p->adjvex;
        //��� visit[w] ��ֵΪ 0����ʾ w �� v �ĺ��ӽ��
        if (visit[w] == 0)
        {
            //������ȱ����������ж� w �Ƿ�Ϊ�ؽ�㣬����� low[w] ��ֵ
            DFSArticul(G, w);
            //��� w ����Ӧ�� low ֵ�� min ֵС������ w �� v ��ĳ�����Ƚ�㽨�������ӣ��� min ������¼�� low[w] ��ֵ
            if (low[w] < min) {
                min = low[w];
            }
            // ��� v ���ķ��ʴ���� <= ���ӽ���¼�����Ƚ��ķ��ʴ���ţ��������ӽ�㲢û�к� v �����ڽ�㽨�����ӣ���ʱ v ���ǹؽ��
            if (visit[v] <= low[w]) {
                // v �����ж�����ӽ�㣬visit[v] <= low[w] �������ܳ�����Ρ�v �ǹؽ�����Ϣֻ��Ҫ���һ�μ��ɡ�
                if (isAritcul == false) {
                    isAritcul = true;
                    printf("%c ", G.vertices[v].data);
                }
            }
        }
        //��� w ֮ǰ���ʹ������� w �� v �����Ƚ�㣬ֱ�Ӹ��� min ��¼�Ĵ����
        else if(visit[w] < min)
        {
            min = visit[w];
        }
    }
    //min ������ֵ������θ��£����ռ�¼���Ǻ� v ��㣨���亢�ӽ�㣩���������Ƚ��ķ��ʴ���ţ�����ж�����Ƚ�㣬min ��¼����ֵ��С�Ĵ����
    low[v] = min;
}

void FindArticul(ALGraph G) {
    //�� G.vertices[0] Ϊ�������������������
    ArcNode* p = G.vertices[0].firstarc;
    visit[0] = 1;
    printf("�ؽڵ��ǣ�\n");
    //����������ĵ�һ�����ӽ�������������������������ҹؽ��
    DFSArticul(G, p->adjvex);
    //����ӵ�һ�����ӽ�����������ܱ��������еĶ��㣬����������ֹ��һ�����ӽ�㣬�������ǹؽ��
    if (count < G.vexnum) {
        printf("%c ", G.vertices[0].data);
        //�����жϱ�����������ҹؽ��
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
