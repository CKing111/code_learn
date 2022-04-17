#



#����һ�����飬����������������Ԫ�ظ���

#����ö��

def L(nums,i):
    """������ʼ��i�ĵ��������г��ȸ�����numΪ�������飬iΪ��ʼ����"""

    if i == len(nums) - 1:   #last number
        return 1

    max_len = 1     #��ʼ�����������
    for j in range (i + 1, len(nums)):      #����0��ʼ����
        if nums[j] > nums[i]:
            max_len = max(max_len,L(nums,j)+1)      #��������nums����ֵ���ʼiֵ���бȽϣ���С�ڳ�ʼʱ�����������ڳ�ʼʱ���������1
        return max_len

def Length_of_LST(nums):
    max_len = 1
    for i in range(len(nums)):
        max_len = max(max_len,L(nums,i)+1)             #������������������
    return max_len

nums = [1,5,2,4,3]

print(Length_of_LST(nums))


#���⸴�ӶȽϸ�


# ���ö�̬�滮��ͨ�������ظ��ڵ�����㣬�����������������
# �����ֵ䡢��ϣ�����洢�м�������
# ���仯�������ÿռ任ʱ�䣬Ҳ�ƴ�����¼�ĵݹ顢�ݹ����ļ�֦(pruning)


memo = {}       #����һ���ֵ䣬����¼�����м�������Ľ��
def L(nums,i):
    """������ʼ��i�ĵ��������г��ȸ�����numΪ�������飬iΪ��ʼ����"""

    if i in memo:
        return memo[i]

    if i == len(nums) - 1:   #last number
        return 1

    max_len = 1     #��ʼ�����������
    for j in range (i + 1, len(nums)):      #����0��ʼ����
        if nums[j] > nums[i]:
            max_len = max(max_len,L(nums,j)+1)      #��������nums����ֵ���ʼiֵ���бȽϣ���С�ڳ�ʼʱ�����������ڳ�ʼʱ���������1
    memo[i] = max_len
    return max_len

def Length_of_LST(nums):
    max_len = 0
    for i in range(len(nums)):
        max_len = max(max_len,L(nums,i))             #������������������
    return max_len

nums = [1,5,2,4,3]

print(Length_of_LST(nums))


#�ǵݹ鷽��������(iterative)������ʽ

"""
[1,5,2,4,3]
���㹫ʽ����1�����Դ˼�����������
L(0) = max{L(1),L(2),L(3),L(4)} + 1
L(1) = max{L(1),L(2),L(3)} + 1      #��5����
L(2) = max{L(1),L(2)} + 1
L(3) = max{L(1)} + 1
L(4) = 1

�Ӻ���ǰ�Դ˼��㣬��������ѧ������
ʵ�֣�
"""

def Length_of_LST(nums):
    n = len(nums)   #5
    L = [1] * n     #��ʼ���б�L:[1��1��1��1��1]

    for i in reversed(range(n)):          #ѭ���У���L(4) -> L(0)
        for j in range(i+1,n):            #ѭ���У���0 -> max{L(1),L(2),L(3),L(4)}
            if nums[j] > nums[i]:         #���ɵ���������
                L[i] = max(L[i],L[j] +1)

    return max(L)

    #��ʽ���Ӷ�ΪO(N^2)С��֮ǰΪO(2^N)


#��Ŀ���ҳ����������е�����
"""
�磺[3,-4,2,-1,2,6,-5,4]����������к�Ϊ[2,-1,2,6]��Ϊ9
"""

#˼·��