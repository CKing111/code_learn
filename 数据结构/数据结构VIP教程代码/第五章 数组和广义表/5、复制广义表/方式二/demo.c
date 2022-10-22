#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int tag;//��־��
    union {
        char atom;//ԭ�ӽ���ֵ��
        struct Node* hp;//�ӱ����ָ����hpָ���ͷ
    }un;
    struct Node* tp;//�����tp�൱�������nextָ�룬����ָ����һ������Ԫ��
}GLNode, * Glist;

Glist creatGlist(Glist C) {
    C = (Glist)malloc(sizeof(GLNode));
    C->tag = 1;
    C->un.hp = (Glist)malloc(sizeof(GLNode));
    C->tp = NULL;
    //�洢 'a'
    C->un.hp->tag = 0;
    C->un.hp->un.atom = 'a';
    //�洢(b,c,d)
    C->un.hp->tp = (Glist)malloc(sizeof(GLNode));
    C->un.hp->tp->tag = 1;
    C->un.hp->tp->un.hp = (Glist)malloc(sizeof(GLNode));
    C->un.hp->tp->tp = NULL;
    //�洢'b'
    C->un.hp->tp->un.hp->tag = 0;
    C->un.hp->tp->un.hp->un.atom = 'b';
    C->un.hp->tp->un.hp->tp = (Glist)malloc(sizeof(GLNode));
    //�洢'c'
    C->un.hp->tp->un.hp->tp->tag = 0;
    C->un.hp->tp->un.hp->tp->un.atom = 'c';
    C->un.hp->tp->un.hp->tp->tp = (Glist)malloc(sizeof(GLNode));
    //�洢'd'
    C->un.hp->tp->un.hp->tp->tp->tag = 0;
    C->un.hp->tp->un.hp->tp->tp->un.atom = 'd';
    C->un.hp->tp->un.hp->tp->tp->tp = NULL;
    return C;
}

void copyGlist(Glist C, Glist* T) {
    //��� C Ϊ NULL��ֱ�ӷ���
    if (C) {
        //���Ʊ�ͷ
        (*T) = (Glist)malloc(sizeof(GLNode));
        if (!*T) {
            exit(0);
        }
        (*T)->tag = C->tag;
        //�����ǰ��������ԭ�ӽ�㣬ֱ�Ӹ�ֵ
        if (C->tag == 0) {
            (*T)->un.atom = C->un.atom;
        }
        else
        {
            //�����ǰ���������ӱ������ݹ鸴������ӱ�
            copyGlist(C->un.hp, &((*T)->un.hp));
        }

        //���Ʊ�β
        if (C->tp == NULL) {
            (*T)->tp = C->tp;
        }
        else
        {
            copyGlist(C->tp, &((*T)->tp));
        }
    }
}

void priGlist(Glist C) {
    Glist cur = C;
    while (cur != NULL) {
        //�����ԭ�ӣ�ֱ�����
        if (cur->tag == 0) {
            printf("%c", cur->un.atom);
        }
        //����ǹ��������ӱ��ݹ������ͷ�ͱ�β
        else
        {
            printf("(");
            priGlist(cur->un.hp);
            printf(")");
        }
        if (cur->tp != NULL) {
            printf(",");
        }
        cur = cur->tp;
    }
}

//�ݹ��ͷŹ����ռ�õĶѿռ�
void delGlist(Glist C) {
    Glist cur = C;
    while (cur != NULL) {
        Glist del = cur;
        if (cur->tag == 1) {
            delGlist(cur->un.hp);
        }
        cur = cur->tp;
        free(del);
    }
}

int main(int argc, const char* argv[]) {
    Glist C = NULL, T = NULL;
    C = creatGlist(C);
    printf("C=");
    priGlist(C);
    copyGlist(C, &T);
    printf("\nT=");
    priGlist(T);
    //ɾ�� C �� T
    delGlist(C);
    delGlist(T);
    system("pause");
    return 0;
}
