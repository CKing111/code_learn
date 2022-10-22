#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int tag;//��־��
    union {
        char atom;//ԭ�ӽ���ֵ��
        struct {
            struct Node* hp, * tp;
        }ptr;//�ӱ����ָ����hpָ���ͷ��tpָ���β
    }un;
}GLNode, * Glist;

Glist creatGlist(Glist C) {
    //�����C
    C = (Glist)malloc(sizeof(GLNode));
    C->tag = 1;
    //��ͷԭ�ӡ�a��
    C->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.hp->tag = 0;
    C->un.ptr.hp->un.atom = 'a';
    //��β�ӱ�(b,c,d),��һ������
    C->un.ptr.tp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->tag = 1;
    C->un.ptr.tp->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.tp = NULL;
    //��ʼ�����һ������Ԫ��(b,c,d),��ͷΪ��b������βΪ(c,d)
    C->un.ptr.tp->un.ptr.hp->tag = 1;
    //�洢 'b'
    C->un.ptr.tp->un.ptr.hp->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.hp->tag = 0;
    C->un.ptr.tp->un.ptr.hp->un.ptr.hp->un.atom = 'b';
    //����ӱ�(c,d)����ͷΪc����βΪ(d)
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->tag = 1;
    //�洢ԭ�� 'c'
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.hp->tag = 0;
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.hp->un.atom = 'c';
    //��ű�β(d)
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->tag = 1;
    //�洢 'd'
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->un.ptr.hp->tag = 0;
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->un.ptr.hp->un.atom = 'd';
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->un.ptr.tp = NULL;
    return C;
}

void copyGlist(Glist C, Glist* T) {
    //���CΪ�ձ���ô���Ʊ�ֱ��Ϊ�ձ�
    if (!C) {
        *T = NULL;
    }
    else {
        *T = (Glist)malloc(sizeof(GLNode));//C���ǿձ���T�����ڴ�ռ�
        //����ʧ�ܣ�����ֹͣ
        if (!*T) {
            exit(0);
        }
        (*T)->tag = C->tag;//���Ʊ�C��tagֵ
        //�жϵ�ǰ��Ԫ���Ƿ�Ϊԭ�ӣ�����ǣ�ֱ�Ӹ���
        if (C->tag == 0) {
            (*T)->un.atom = C->un.atom;
        }
        else {//���е��⣬˵�����Ƶ����ӱ�
            copyGlist(C->un.ptr.hp, &((*T)->un.ptr.hp));//���Ʊ�ͷ
            copyGlist(C->un.ptr.tp, &((*T)->un.ptr.tp));//���Ʊ�β
        }
    }
}

//�ݹ�ɾ�������ռ�õĶ��ڴ�
void delGlist(Glist C) {
    if (C != NULL) {
        //������ӱ��㣬������������ hp �� tp ����ֱ������ hp �� tp ����û�п��ͷŵĶѿռ�֮���ͷŵ�ǰ���Ŀռ�
        if (C->tag == 1) {
            delGlist(C->un.ptr.hp);
            delGlist(C->un.ptr.tp);
            free(C);
            return;
        }
        //ɾ��ԭ�ӽ��
        else
        {
            free(C);
            return;
        }
    }
}
//�ݹ����������е��ӱ�
void pri_SubGlist(Glist C) {
    if (C) {
        //�����ͷ���� hp Ϊ NULL����ʾ����һ���ձ�
        if (!C->un.ptr.hp) {
            printf("()");
        }
        else {
            //�����ͷ��ԭ�ӣ�ֱ�����
            if (C->un.ptr.hp->tag == 0) {
                putchar(C->un.ptr.hp->un.atom);
            }
            //�����ͷ���ӱ������������ӱ�
            else
            {
                putchar('(');
                pri_SubGlist(C->un.ptr.hp);
                putchar(')');
            }
        }
        //�����β���ǿձ���ͬ���ķ�ʽ�����β
        if (C->un.ptr.tp) {
            putchar(',');
            pri_SubGlist(C->un.ptr.tp);
        }
    }
}
//��������
void priGlist(Glist C) {
    putchar('(');
    pri_SubGlist(C);
    putchar(')');
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
    return 0;
}
