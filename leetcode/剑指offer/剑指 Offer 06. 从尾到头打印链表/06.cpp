/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> res;
    vector<int> reversePrint(ListNode* head) {
        //方法1：reverse反转法
        /*
        while(head){
            res.push_back(head->val);
            head = head->next;
        }
        //使用algorithm算法中的reverse反转res
        reverse(res.begin(),res.end());
        return res;
        */

        //方法2：入栈法
        /*
        stack<int> s;       //声明栈s
        //入栈
        while(head){
            s.push(head->val);  //入栈
            head = head->next;  //后移指针
        }
        //出栈
        while(!s.empty()){
            res.push_back(s.top());     //res中压入s的头元素
            s.pop();    //s出栈
        }
        return res;
        */

        //方法3：递归
        /*
        if(head == nullptr) //为空
            return res;
        reversePrint(head->next);
        res.push_back(head->val);
        return res;
        */

        //方法4：改变链表结构
        ListNode *pre = nullptr;
        ListNode *next = head;
        ListNode *cur = head;
        while(cur){
            next = cur->next;//保存当前结点的下一个节点
            cur->next = pre;//当前结点指向前一个节点，反向改变指针
            pre = cur;//更新前一个节点
            cur = next;//更新当前结点
        }
        while(pre){//上一个while循环结束后，pre指向新的链表头
            res.push_back(pre->val);
            pre = pre->next;
        }
        return res;
    }
};
