#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int tag;//标志域
    union {
        char atom;//原子结点的值域
        struct {
            struct Node* hp, * tp;
        }ptr;//子表结点的指针域，hp指向表头；tp指向表尾
    }un;
}GLNode, * Glist;

Glist creatGlist(Glist C) {
    //广义表C
    C = (Glist)malloc(sizeof(GLNode));
    C->tag = 1;
    //表头原子‘a’
    C->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.hp->tag = 0;
    C->un.ptr.hp->un.atom = 'a';
    //表尾子表(b,c,d),是一个整体
    C->un.ptr.tp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->tag = 1;
    C->un.ptr.tp->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.tp = NULL;
    //开始存放下一个数据元素(b,c,d),表头为‘b’，表尾为(c,d)
    C->un.ptr.tp->un.ptr.hp->tag = 1;
    //存储 'b'
    C->un.ptr.tp->un.ptr.hp->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.hp->tag = 0;
    C->un.ptr.tp->un.ptr.hp->un.ptr.hp->un.atom = 'b';
    //存放子表(c,d)，表头为c，表尾为(d)
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->tag = 1;
    //存储原子 'c'
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.hp->tag = 0;
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.hp->un.atom = 'c';
    //存放表尾(d)
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->tag = 1;
    //存储 'd'
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->un.ptr.hp = (Glist)malloc(sizeof(GLNode));
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->un.ptr.hp->tag = 0;
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->un.ptr.hp->un.atom = 'd';
    C->un.ptr.tp->un.ptr.hp->un.ptr.tp->un.ptr.tp->un.ptr.tp = NULL;
    return C;
}

void copyGlist(Glist C, Glist* T) {
    //如果C为空表，那么复制表直接为空表
    if (!C) {
        *T = NULL;
    }
    else {
        *T = (Glist)malloc(sizeof(GLNode));//C不是空表，给T申请内存空间
        //申请失败，程序停止
        if (!*T) {
            exit(0);
        }
        (*T)->tag = C->tag;//复制表C的tag值
        //判断当前表元素是否为原子，如果是，直接复制
        if (C->tag == 0) {
            (*T)->un.atom = C->un.atom;
        }
        else {//运行到这，说明复制的是子表
            copyGlist(C->un.ptr.hp, &((*T)->un.ptr.hp));//复制表头
            copyGlist(C->un.ptr.tp, &((*T)->un.ptr.tp));//复制表尾
        }
    }
}

//递归删除广义表占用的堆内存
void delGlist(Glist C) {
    if (C != NULL) {
        //如果是子表结点，继续查找它的 hp 和 tp 方向，直至它的 hp 和 tp 方向都没有可释放的堆空间之后，释放当前结点的空间
        if (C->tag == 1) {
            delGlist(C->un.ptr.hp);
            delGlist(C->un.ptr.tp);
            free(C);
            return;
        }
        //删除原子结点
        else
        {
            free(C);
            return;
        }
    }
}
//递归输出广义表中的子表
void pri_SubGlist(Glist C) {
    if (C) {
        //如果表头结点的 hp 为 NULL，表示这是一个空表
        if (!C->un.ptr.hp) {
            printf("()");
        }
        else {
            //如果表头是原子，直接输出
            if (C->un.ptr.hp->tag == 0) {
                putchar(C->un.ptr.hp->un.atom);
            }
            //如果表头是子表，继续遍历该子表
            else
            {
                putchar('(');
                pri_SubGlist(C->un.ptr.hp);
                putchar(')');
            }
        }
        //如果表尾不是空表，以同样的方式处理表尾
        if (C->un.ptr.tp) {
            putchar(',');
            pri_SubGlist(C->un.ptr.tp);
        }
    }
}
//输出广义表
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

    //删除 C 和 T
    delGlist(C);
    delGlist(T);
    return 0;
}
