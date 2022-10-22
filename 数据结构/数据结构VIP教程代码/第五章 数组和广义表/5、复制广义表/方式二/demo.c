#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int tag;//标志域
    union {
        char atom;//原子结点的值域
        struct Node* hp;//子表结点的指针域，hp指向表头
    }un;
    struct Node* tp;//这里的tp相当于链表的next指针，用于指向下一个数据元素
}GLNode, * Glist;

Glist creatGlist(Glist C) {
    C = (Glist)malloc(sizeof(GLNode));
    C->tag = 1;
    C->un.hp = (Glist)malloc(sizeof(GLNode));
    C->tp = NULL;
    //存储 'a'
    C->un.hp->tag = 0;
    C->un.hp->un.atom = 'a';
    //存储(b,c,d)
    C->un.hp->tp = (Glist)malloc(sizeof(GLNode));
    C->un.hp->tp->tag = 1;
    C->un.hp->tp->un.hp = (Glist)malloc(sizeof(GLNode));
    C->un.hp->tp->tp = NULL;
    //存储'b'
    C->un.hp->tp->un.hp->tag = 0;
    C->un.hp->tp->un.hp->un.atom = 'b';
    C->un.hp->tp->un.hp->tp = (Glist)malloc(sizeof(GLNode));
    //存储'c'
    C->un.hp->tp->un.hp->tp->tag = 0;
    C->un.hp->tp->un.hp->tp->un.atom = 'c';
    C->un.hp->tp->un.hp->tp->tp = (Glist)malloc(sizeof(GLNode));
    //存储'd'
    C->un.hp->tp->un.hp->tp->tp->tag = 0;
    C->un.hp->tp->un.hp->tp->tp->un.atom = 'd';
    C->un.hp->tp->un.hp->tp->tp->tp = NULL;
    return C;
}

void copyGlist(Glist C, Glist* T) {
    //如果 C 为 NULL，直接返回
    if (C) {
        //复制表头
        (*T) = (Glist)malloc(sizeof(GLNode));
        if (!*T) {
            exit(0);
        }
        (*T)->tag = C->tag;
        //如果当前遍历的是原子结点，直接赋值
        if (C->tag == 0) {
            (*T)->un.atom = C->un.atom;
        }
        else
        {
            //如果当前遍历的是子表，继续递归复制这个子表
            copyGlist(C->un.hp, &((*T)->un.hp));
        }

        //复制表尾
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
        //如果是原子，直接输出
        if (cur->tag == 0) {
            printf("%c", cur->un.atom);
        }
        //如果是广义表或者子表，递归输出表头和表尾
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

//递归释放广义表占用的堆空间
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
    //删除 C 和 T
    delGlist(C);
    delGlist(T);
    system("pause");
    return 0;
}
