#include<stdlib.h>
#include<stdio.h>
#include<string.h>
//�����������ṹ
typedef struct {
    int weight;//���Ȩ��
    int parent, left, right;//����㡢���ӡ��Һ����������е�λ���±�
}HTNode, * HuffmanTree;

//��̬��ά���飬�洢����������
typedef char** HuffmanCode;

//HT�����д�ŵĹ���������end��ʾHT�����д�Ž�������λ�ã�s1��s2���ݵ���HT������Ȩ��ֵ��С����������������е�λ��
void Select(HuffmanTree HT, int end, int* s1, int* s2)
{
    int min1, min2;
    int i = 1, j;
    //�ҵ���û�������Ľ��
    while (HT[i].parent != 0 && i <= end) {
        i++;
    }
    min1 = HT[i].weight;
    *s1 = i;

    i++;
    while (HT[i].parent != 0 && i <= end) {
        i++;
    }
    //���ҵ����������Ƚϴ�С��min2Ϊ��ģ�min1ΪС��
    if (HT[i].weight < min1) {
        min2 = min1;
        *s2 = *s1;
        min1 = HT[i].weight;
        *s1 = i;
    }
    else {
        min2 = HT[i].weight;
        *s2 = i;
    }
    //�������ͺ���������δ���������Ľ�����Ƚ�
    for (j = i + 1; j <= end; j++)
    {
        //����и���㣬ֱ��������������һ��
        if (HT[j].parent != 0) {
            continue;
        }
        //�������С�Ļ�С����min2=min1��min1��ֵ�µĽ����±�
        if (HT[j].weight < min1) {
            min2 = min1;
            min1 = HT[j].weight;
            *s2 = *s1;
            *s1 = j;
        }
        //�����������֮�䣬min2��ֵΪ�µĽ���λ���±�
        else if (HT[j].weight >= min1 && HT[j].weight < min2) {
            min2 = HT[j].weight;
            *s2 = j;
        }
    }
}

//HTΪ��ַ���ݵĴ洢�������������飬wΪ�洢���Ȩ��ֵ�����飬nΪ������
void CreateHuffmanTree(HuffmanTree* HT, int* w, int n)
{
    int m, i;
    if (n <= 1) return; // ���ֻ��һ��������൱��0
    m = 2 * n - 1; // ���������ܽڵ�����n����Ҷ�ӽ��
    *HT = (HuffmanTree)malloc((m + 1) * sizeof(HTNode)); // 0��λ�ò���
    HuffmanTree p = *HT;
    // ��ʼ�����������е����н��
    for (i = 1; i <= n; i++)
    {
        (p + i)->weight = *(w + i - 1);
        (p + i)->parent = 0;
        (p + i)->left = 0;
        (p + i)->right = 0;
    }
    //��������±� n+1 ��ʼ��ʼ�����������г�Ҷ�ӽ����Ľ��
    for (i = n + 1; i <= m; i++)
    {
        (p + i)->weight = 0;
        (p + i)->parent = 0;
        (p + i)->left = 0;
        (p + i)->right = 0;
    }
    //������������
    for (i = n + 1; i <= m; i++)
    {
        int s1, s2;
        Select(*HT, i - 1, &s1, &s2);
        (*HT)[s1].parent = (*HT)[s2].parent = i;
        (*HT)[i].left = s1;
        (*HT)[i].right = s2;
        (*HT)[i].weight = (*HT)[s1].weight + (*HT)[s2].weight;
    }
}
//HTΪ����������HCΪ�洢������������Ķ�ά��̬���飬nΪ���ĸ���
void HuffmanCoding(HuffmanTree HT, HuffmanCode* HC, int n) {
    int i, m, p, cdlen;
    char* cd = NULL;
    *HC = (HuffmanCode)malloc((n + 1) * sizeof(char*));
    memset(*HC, NULL, n + 1);
    m = 2 * n - 1;
    p = m;
    cdlen = 0;
    cd = (char*)malloc(n * sizeof(char));
    memset(cd, 0, n);
    //����������Ȩ�����ڼ�¼���ʽ��Ĵ��������ȳ�ʼ��Ϊ0
    for (i = 1; i <= m; i++) {
        HT[i].weight = 0;
    }
    //һ��ʼ p ��ʼ��Ϊ m��Ҳ���Ǵ�������ʼ��һֱ��pΪ0
    while (p) {
        //�����ǰ���һ��û�з��ʣ��������if���
        if (HT[p].weight == 0) {
            HT[p].weight = 1;//���÷��ʴ���Ϊ1
            //��������ӣ���������ӣ����Ҵ洢�߹��ı��Ϊ0
            if (HT[p].left != 0) {
                p = HT[p].left;
                cd[cdlen++] = '0';
            }
            //��ǰ���û�����ӣ�Ҳû���Һ��ӣ�˵��ΪҶ�ӽ�㣬ֱ�Ӽ�¼����������
            else if (HT[p].right == 0) {
                (*HC)[p] = (char*)malloc((cdlen + 1) * sizeof(char));
                cd[cdlen] = '\0';
                strcpy((*HC)[p], cd);
            }
        }
        //���weightΪ1��˵�����ʹ�һ�Σ����Ǵ������ӷ��ص�
        else if (HT[p].weight == 1) {
            HT[p].weight = 2;//���÷��ʴ���Ϊ2
            //������Һ��ӣ������Һ��ӣ���¼���ֵ 1
            if (HT[p].right != 0) {
                p = HT[p].right;
                cd[cdlen++] = '1';
            }
        }
        //������ʴ���Ϊ 2��˵�����Һ��Ӷ��������ˣ����ظ����
        else {
            HT[p].weight = 0;
            p = HT[p].parent;
            --cdlen;
        }
    }
    free(cd);
}
//��ӡ����������ĺ���
void PrintHuffmanCode(HuffmanCode htable, int* w, int n)
{
    printf("Huffman code : \n");
    for (int i = 1; i <= n; i++) {
        printf("%d code = %s\n", w[i - 1], htable[i]);
    }

}

void DelHuffmanCode(HuffmanCode htable, int n) {
    int i;
    for (i = 0; i < n + 1; i++) {
        if (htable[i]) {
            free(htable[i]);
        }
    }
    free(htable);
}

int main(void)
{
    int w[5] = { 2, 8, 7, 6, 5 };
    int n = 5;
    HuffmanTree htree;
    HuffmanCode htable;
    CreateHuffmanTree(&htree, w, n);
    HuffmanCoding(htree, &htable, n);
    PrintHuffmanCode(htable, w, n);
    //�ͷ�����Ķ��ڴ�
    free(htree);
    DelHuffmanCode(htable, n);
    system("pause");
    return 0;
}
