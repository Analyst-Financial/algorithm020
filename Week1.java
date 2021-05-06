package algorithm;

import java.util.*;

/**
 * @ClassName Week1
 * @Description
 * @Author Administrator
 * @Date 2020/11/22  5:09
 * @Version 1.0
 **/
public class Week1 {
    /**
     * 链表相关文章汇总
     * 题号	题目	题解	难度等级
     * 19	删除链表的倒数第N个节点	两种实现+图解	中等
     * 21	合并两个有序链表	两种实现+图解	简单
     * 23	合并K个升序链表	四种实现+图解	困难
     * 24	两两交换链表中的节点	三种实现+图解	中等
     * 25	K 个一组翻转链表	两种实现+图解	困难
     * 61	旋转链表	两种实现+图解	中等
     * 82	删除排序链表中的重复元素 II	三种实现+图解	中等
     * 83	删除排序链表中的重复元素	两种实现+图解	简单
     * 141	二叉树展开为链表	四种实现+图解	中等
     * 138	复制带随机指针的链表	两种实现+图解	中等
     * 141	环形链表	两种实现+图解	简单
     * 160	相交链表	两种实现+图解	简单
     * 203	移除链表元素	两种实现+图解	简单
     * 206	反转链表	两种实现+图解	简单
     * 234	回文链表	图解	简单
     * 237	删除链表中的节点	图解	简单
     * 876	链表的中间结点	图解	简单
     * *****************************************************************************************************************
     * 第1周 第3课 | 数组、链表、跳表
     * *****************************************************************************************************************
     * 参考链接
     * Java 源码分析（ArrayList）
     * Linked List 的标准实现代码
     * Linked List 示例代码
     * Java 源码分析（LinkedList）
     * LRU Cache - Linked list： LRU 缓存机制
     * Redis - Skip List：跳跃表、为啥 Redis 使用跳表（Skip List）而不是使用 Red-Black？
     * *****************************************************************************************************************
     * Array 实战题目
     * 两数之和（近半年内，字节跳动在面试中考查此题达到 152 次）
     * 盛最多水的容器（腾讯、百度、字节跳动在近半年内面试常考）
     * 移动零（华为、字节跳动在近半年内面试常考）
     * 爬楼梯（阿里巴巴、腾讯、字节跳动在半年内面试常考）
     * 三数之和（国内、国际大厂历年面试高频老题）
     * Linked List 实战题目
     * 反转链表（字节跳动、亚马逊在半年内面试常考）
     * 两两交换链表中的节点（阿里巴巴、字节跳动在半年内面试常考）
     * 环形链表（阿里巴巴、字节跳动、腾讯在半年内面试常考）
     * 环形链表 II
     * K 个一组翻转链表（字节跳动、猿辅导在半年内面试常考）
     * *****************************************************************************************************************
     */

    /**
     * 206. 反转链表（字节跳动、亚马逊在半年内面试常考）
     * 反转一个单链表。
     * *****************************************************************************************************************
     * 示例:
     * 输入: 1->2->3->4->5->NULL
     * 输出: 5->4->3->2->1->NULL
     * 进阶:
     * 你可以迭代或递归地反转链表。你能否用两种方法解决这道题？
     * *****************************************************************************************************************
     * 利用外部空间
     * 这种方式很简单，先申请一个动态扩容的数组或者容器，比如 ArrayList 这样的。
     * 然后不断遍历链表，将链表中的元素添加到这个容器中。
     * 再利用容器自身的 API，反转整个容器，这样就达到反转的效果了。
     * 最后同时遍历容器和链表，将链表中的值改为容器中的值。
     * 因为此时容器的值是：
     * 5 4 3 2 1
     * 链表按这个顺序重新被设置一边，就达到要求啦。
     * 当然你可以可以再新创建 N 个节点，然后再返回，这样也可以达到目的。
     * 这种方式很简单，但你在面试中这么做的话，面试官 100% 会追问是否有更优的方式，比如不用外部空间。所以我就不做代码和动画演示了，下面来看看如何用 O(1)O(1) 空间复杂度来实现这道题。
     *
     * 双指针迭代
     * 我们可以申请两个指针，第一个指针叫 pre，最初是指向 null 的。
     * 第二个指针 cur 指向 head，然后不断遍历 cur。
     * 每次迭代到 cur，都将 cur 的 next 指向 pre，然后 pre 和 cur 前进一位。
     * 都迭代完了(cur 变成 null 了)，pre 就是最后一个节点了。
     * 动画演示如下：
     * 链接：https://leetcode-cn.com/problems/reverse-linked-list/solution/dong-hua-yan-shi-206-fan-zhuan-lian-biao-by-user74/
     * 动画演示中其实省略了一个tmp变量，这个tmp变量会将cur的下一个节点保存起来，考虑到一张动画放太多变量会很混乱，所以我就没加了，具体详细执行过程，请点击下面的幻灯片查看。
     * 链接：https://leetcode-cn.com/problems/reverse-linked-list/solution/dong-hua-yan-shi-206-fan-zhuan-lian-biao-by-user74/
     * 206. Reverse Linked List
     * https://leetcode.com/problems/reverse-linked-list/
     */
    public ListNode reverseList(ListNode head) {
        //申请节点，pre和 cur，pre指向null
        ListNode pre = null;
        ListNode cur = head;
        ListNode tmp = null;
        while (cur != null) {
            //记录当前节点的下一个节点
            tmp = cur.next;
            //然后将当前节点指向pre
            cur.next = pre;
            //pre和cur节点都前进一位
            pre = cur;
            cur = tmp;
        }
        return pre;
    }

    /**
     * 递归解法
     * 这题有个很骚气的递归解法，递归解法很不好理解，这里最好配合代码和动画一起理解。
     * 递归的两个条件：
     *
     * 终止条件是当前节点或者下一个节点==null
     * 在函数内部，改变节点的指向，也就是 head 的下一个节点指向 head 递归函数那句
     *
     * head.next.next = head
     * 很不好理解，其实就是 head 的下一个节点指向head。
     * 递归函数中每次返回的 cur 其实只最后一个节点，在递归函数内部，改变的是当前节点的指向。
     * 动画演示如下：
     * 链接：https://leetcode-cn.com/problems/reverse-linked-list/solution/dong-hua-yan-shi-206-fan-zhuan-lian-biao-by-user74/
     * 幻灯片演示
     * 感谢@zhuuuu-2的建议，递归的解法光看动画比较容易理解，但真到了代码层面理解起来可能会有些困难，我补充了下递归调用的详细执行过程。
     * 以1->2->3->4->5这个链表为例，整个递归调用的执行过程，对应到代码层面(用java做示范)是怎么执行的，以及递归的调用栈都列出来了，请点击下面的幻灯片查看吧。
     * 链接：https://leetcode-cn.com/problems/reverse-linked-list/solution/dong-hua-yan-shi-206-fan-zhuan-lian-biao-by-user74/
     */
    public ListNode reverseList2(ListNode head) {
        //递归终止条件是当前为空，或者下一个节点为空
        if (head == null || head.next == null) {
            return head;
        }
        //这里的cur就是最后一个节点
        ListNode cur = reverseList(head.next);
        //这里请配合动画演示理解
        //如果链表是 1->2->3->4->5，那么此时的cur就是5
        //而head是4，head的下一个是5，下下一个是空
        //所以head.next.next 就是5->4
        head.next.next = head;
        //防止链表循环，需要将head.next设置为空
        head.next = null;
        //每层递归函数都返回cur，也就是最后一个节点
        return cur;
    }

    /**
     * 方法一 迭代
     * 解题思路
     * 通过迭代 将 1->2->3->4->5->∮ 转换成 ∮<-1<-2<-3<-4<-5
     * 如图执行过程
     * https://leetcode-cn.com/problems/reverse-linked-list/solution/die-dai-by-javaniuniu/
     */
    public ListNode reverseList3(ListNode head) {
        // 申请两个链表 一个空链表，一个完整的链表
        ListNode pre = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode temp = curr.next;
            curr.next = pre; // 当前链表指向 新链表
            pre = curr; // 赋值给新链表
            curr = temp;
        }
        return pre;

    }

    /**
     * We always put a node's previous node as one's next,
     * Take 1 -> 2 -> 3 -> N for example, we reverse the list by
     * put 1's previous node null as 1's next,
     * put 2's previous node 1 as 2's next,
     * put 3's previous node 2 as 3's next,
     * return 3 // put null's previous node 3 as null's next
     * The code is as follows:
     */
    public ListNode reverseList4(ListNode head) {
        return putPreAfterNode(head, null);
    }

    private ListNode putPreAfterNode(ListNode node, ListNode pre) {
        if (node == null) {
            return pre;
        }
        ListNode next = node.next;
        node.next = pre;
        return putPreAfterNode(next, node);
    }
    /**
     * 24. 两两交换链表中的节点（阿里巴巴、字节跳动在半年内面试常考）
     * 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
     * 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
     * *****************************************************************************************************************
     * 示例 1：
     *
     * 输入：head = [1,2,3,4]
     * 输出：[2,1,4,3]
     * 示例 2：
     *
     * 输入：head = []
     * 输出：[]
     * 示例 3：
     *
     * 输入：head = [1]
     * 输出：[1]
     * *****************************************************************************************************************
     * 解题思路
     * 标签：链表
     * 本题的递归和非递归解法其实原理类似，都是更新每两个点的链表形态完成整个链表的调整
     * 其中递归解法可以作为典型的递归解决思路进行讲解
     * 递归写法要观察本级递归的解决过程，形成抽象模型，因为递归本质就是不断重复相同的事情。而不是去思考完整的调用栈，一级又一级，无从下手。如图所示，我们应该关注一级调用小单元的情况，也就是单个f(x)。
     * 链接：https://leetcode-cn.com/problems/swap-nodes-in-pairs/solution/hua-jie-suan-fa-24-liang-liang-jiao-huan-lian-biao/
     * 其中我们应该关心的主要有三点：
     * 返回值
     * 调用单元做了什么
     * 终止条件
     * 在本题中：
     * 返回值：交换完成的子链表
     * 调用单元：设需要交换的两个点为 head 和 next，head 连接后面交换完成的子链表，next 连接 head，完成交换
     * 终止条件：head 为空指针或者 next 为空指针，也就是当前无节点或者只有一个节点，无法进行交换
     * *****************************************************************************************************************
     * 代码
     * 递归解法
     * 链接：https://leetcode-cn.com/problems/swap-nodes-in-pairs/solution/hua-jie-suan-fa-24-liang-liang-jiao-huan-lian-biao/
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = head.next;
        head.next = swapPairs(next.next);
        next.next = head;
        return next;
    }

    /**
     * 24. Swap Nodes in Pairs
     * Given a linked list, swap every two adjacent nodes and return its head.
     * 非递归解法
     */
    public ListNode swapPairs2(ListNode head) {
        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode temp = pre;
        while (temp.next != null && temp.next.next != null) {
            ListNode start = temp.next;
            ListNode end = temp.next.next;
            temp.next = end;
            start.next = end.next;
            end.next = start;
            temp = start;
        }
        return pre.next;
    }

    /**
     * 141. Linked List Cycle
     * Given head, the head of a linked list, determine if the linked list has a cycle in it.
     * There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
     * Return true if there is a cycle in the linked list. Otherwise, return false.
     * *****************************************************************************************************************
     * Example 1:
     * Input: head = [3,2,0,-4], pos = 1
     * Output: true
     * Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
     *
     * Example 2:
     * Input: head = [1,2], pos = 0
     * Output: true
     * Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.
     *
     * Example 3:
     * Input: head = [1], pos = -1
     * Output: false
     * Explanation: There is no cycle in the linked list.
     * *****************************************************************************************************************
     * Use two pointers, walker and runner.
     * walker moves step by step. runner moves two steps at time.
     * if the Linked List has a cycle walker and runner will meet at some point.
     */
    public boolean hasCycle(ListNode head) {
        if (head == null) return false;
        ListNode walker = head;
        ListNode runner = head;
        while (runner.next != null && runner.next.next != null) {
            walker = walker.next;
            runner = runner.next.next;
            if (walker == runner) return true;
        }
        return false;
    }

    /**
     * 141. 环形链表
     * 给定一个链表，判断链表中是否有环。
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
     * 如果链表中存在环，则返回 true 。 否则，返回 false 。
     * 进阶：
     * 你能用 O(1)（即，常量）内存解决此问题吗？
     * https://leetcode-cn.com/problems/linked-list-cycle/solution/141-huan-xing-lian-biao-shuang-zhi-zhen-4uc9q/
     * *****************************************************************************************************************
     * The if(head==null) check can be further removed.
     * star 18179 Reputation
     */
    public boolean hasCycle2(ListNode head) {
        ListNode walker = head;
        ListNode runner = head;
        while (runner != null && runner.next != null) {
            walker = walker.next;
            runner = runner.next.next;
            if (walker == runner) return true;
        }
        return false;
    }

    /**
     * 思路
     * 可以使用快慢指针法， 分别定义 fast 和 slow指针，从头结点出发，fast指针每次移动两个节点，slow指针每次移动一个节点，如果 fast 和 slow指针在途中相遇 ，说明这个链表有环。
     * 为什么fast 走两个节点，slow走一个节点，有环的话，一定会在环内相遇呢，而不是永远的错开呢？
     * 首先第一点： fast指针一定先进入环中，如果fast 指针和slow指针相遇的话，一定是在环中相遇，这是毋庸置疑的。
     * 那么来看一下，为什么fast指针和slow指针一定会相遇呢？
     * 可以画一个环，然后让 fast指针在任意一个节点开始追赶slow指针。
     * 会发现最终都是这种情况， 如下图：
     * 链接：https://leetcode-cn.com/problems/linked-list-cycle/solution/141-huan-xing-lian-biao-shuang-zhi-zhen-4uc9q/
     * fast和slow各自再走一步， fast和slow就相遇了
     * 这是因为fast是走两步，slow是走一步，其实相对于slow来说，fast是一个节点一个节点的靠近slow的，所以fast一定可以和slow重合。
     * 动画如下：
     * 链接：https://leetcode-cn.com/problems/linked-list-cycle/solution/141-huan-xing-lian-biao-shuang-zhi-zhen-4uc9q/
     * C++代码如下
     * class Solution {
     * public:
     *     bool hasCycle(ListNode *head) {
     *         ListNode* fast = head;
     *         ListNode* slow = head;
     *         while(fast != NULL && fast->next != NULL) {
     *             slow = slow->next;
     *             fast = fast->next->next;
     *             // 快慢指针相遇，说明有环
     *             if (slow == fast) return true;
     *         }
     *         return false;
     *     }
     * };
     * 扩展
     * 做完这道题目，可以在做做142.环形链表II，不仅仅要找环，还要找环的入口。
     * 142.环形链表II题解：链表：环找到了，那入口呢？
     * 我已经陆续将我的题解按照由浅入深的刷题顺序编排起来，整理成册，这份刷题顺序和题解在公众号里已经陪伴了上万录友。
     * PDF中不仅有刷题大纲、刷题顺序，还有详细图解，每一本PDF发布之后都广受好评，这也是Carl花费大量时间写题解的动力。
     * 链接：https://leetcode-cn.com/problems/linked-list-cycle/solution/141-huan-xing-lian-biao-shuang-zhi-zhen-4uc9q/
     */

    /**
     * 142. Linked List Cycle II
     * Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
     * There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
     * Notice that you should not modify the linked list.
     * *****************************************************************************************************************
     * Example 1:
     * Input: head = [3,2,0,-4], pos = 1
     * Output: tail connects to node index 1
     * Explanation: There is a cycle in the linked list, where tail connects to the second node.
     *
     * Example 2:
     * Input: head = [1,2], pos = 0
     * Output: tail connects to node index 0
     * Explanation: There is a cycle in the linked list, where tail connects to the first node.
     *
     * Example 3:
     * Input: head = [1], pos = -1
     * Output: no cycle
     * Explanation: There is no cycle in the linked list.
     * *****************************************************************************************************************
     * https://leetcode.com/problems/linked-list-cycle-ii/
     */
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;

            if (fast == slow) {
                ListNode slow2 = head;
                while (slow2 != slow) {
                    slow = slow.next;
                    slow2 = slow2.next;
                }
                return slow;
            }
        }
        return null;
    }

    /**
     * 142. 环形链表 II
     * 给定一个链表，返回链表开始入环的第一个节点。如果链表无环，则返回null。
     * 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
     * 说明：不允许修改给定的链表。
     * 进阶：
     * 你是否可以使用 O(1) 空间解决此题？
     * *****************************************************************************************************************
     * Python:
     * class Solution(object):
     *     def detectCycle(self, head):
     *         fast, slow = head, head
     *         while True:
     *             if not (fast and fast.next): return
     *             fast, slow = fast.next.next, slow.next
     *             if fast == slow: break
     *         fast = head
     *         while fast != slow:
     *             fast, slow = fast.next, slow.next
     *         return fast
     *
     * 链接：https://leetcode-cn.com/problems/linked-list-cycle-ii/solution/linked-list-cycle-ii-kuai-man-zhi-zhen-shuang-zhi-/
     * *****************************************************************************************************************
     * 环形链表 II（双指针法，清晰图解）
     * 解题思路：
     * 这类链表题目一般都是使用双指针法解决的，例如寻找距离尾部第K个节点、寻找环入口、寻找公共尾部入口等。
     * 算法流程：
     * 1.双指针第一次相遇： 设两指针 fast，slow 指向链表头部 head，fast 每轮走 2 步，slow 每轮走 1 步；
     *  第一种结果： fast 指针走过链表末端，说明链表无环，直接返回 null；
     *  TIPS: 若有环，两指针一定会相遇。因为每走 1 轮，fast 与 slow 的间距 +1，fast 终会追上 slow；
     *  第二种结果： 当fast == slow时， 两指针在环中 第一次相遇 。下面分析此时fast 与 slow走过的 步数关系 ：
     *  设链表共有 a+b 个节点，其中 链表头部到链表入口 有 a 个节点（不计链表入口节点）， 链表环 有 b 个节点（这里需要注意，a 和 b 是未知数，例如图解上链表 a=4 , b=5）；设两指针分别走了 f，s 步，则有：
     *  fast 走的步数是slow步数的 2 倍，即 f = 2s；（解析： fast 每轮走 2 步）
     *  fast 比 slow多走了 n 个环的长度，即 f = s + nb；（ 解析： 双指针都走过 a 步，然后在环内绕圈直到重合，重合时 fast 比 slow 多走 环的长度整数倍 ）；
     *  以上两式相减得：f = 2nb，s = nb，即fast和slow 指针分别走了 2n，n 个 环的周长 （注意： n 是未知数，不同链表的情况不同）。
     * 2.目前情况分析：
     *  如果让指针从链表头部一直向前走并统计步数k，那么所有 走到链表入口节点时的步数 是：k=a+nb（先走 a 步到入口节点，之后每绕 1 圈环（ b 步）都会再次到入口节点）。
     *  而目前，slow 指针走过的步数为 nb 步。因此，我们只要想办法让 slow 再走 a 步停下来，就可以到环的入口。
     *  但是我们不知道 a 的值，该怎么办？依然是使用双指针法。我们构建一个指针，此指针需要有以下性质：此指针和slow 一起向前走 a 步后，两者在入口节点重合。那么从哪里走到入口节点需要 aa 步？答案是链表头部head。
     * 3.双指针第二次相遇：
     *  slow指针 位置不变 ，将fast指针重新 指向链表头部节点 ；slow和fast同时每轮向前走 1 步；
     *  TIPS：此时 f = 0，s = nb ；
     *  当 fast 指针走到f = a 步时，slow 指针走到步s = a+nb ，此时 两指针重合，并同时指向链表环入口 。
     *  返回slow指针指向的节点。
     *
     * 复杂度分析：
     * 时间复杂度 O(N) ：第二次相遇中，慢指针须走步数 a < a + b ；第一次相遇中，慢指针须走步数 a + b - x < a + b ，其中 x 为双指针重合点与环入口距离；因此总体为线性复杂度；
     * 空间复杂度 O(1) ：双指针使用常数大小的额外空间。
     *
     * 我觉得讲的很清楚了 get到的关键点是：
     * 1.走a+nb步一定是在环入口
     * 2.第一次相遇时慢指针已经走了nb步
     * 链接：https://leetcode-cn.com/problems/linked-list-cycle-ii/solution/linked-list-cycle-ii-kuai-man-zhi-zhen-shuang-zhi-/
     */
    public ListNode detectCycle2(ListNode head) {
        ListNode fast = head, slow = head;
        while (true) {
            if (fast == null || fast.next == null) return null;
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) break;
        }
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return fast;
    }

    /**
     * 环形链表 II（哈希表 / 快慢指针（再看）☀
     * 方法一：哈希表（存储第一次出现的节点）
     * 哈希表：从头开始遍历链表，若当前遍历的节点没有出现过，则加入到哈希表中；若第一次出现当前节点在哈希表中出现过，则该节点一定是环的开始，返回即可。
     *
     * 链接：https://leetcode-cn.com/problems/linked-list-cycle-ii/solution/huan-xing-lian-biao-iiha-xi-biao-kuai-ma-ki2r/
     */
    public ListNode detectCycle3(ListNode head) {
        Set<ListNode> set = new HashSet<>();
        while (head != null) {
            if (set.contains(head)) {
                return head;
            } else {
                set.add(head);
                head = head.next;
            }
        }
        return null;
    }

    /**
     * 方法二：快慢指针（看看看）
     * 首先判断链表中是否存在环，利用快慢指针：fast 指针一次走两步，slow 指针一次走一步。
     * 若链表中存在环，则 fast 指针和 slow 指针一定会相遇，即 fast == slow，此时令其中一个指针重新指向头节点 head 并且两个指针每次只走一步，则当两个指针再次相遇时，一定为环的开始；否则，返回 null 即可。
     * 链接：https://leetcode-cn.com/problems/linked-list-cycle-ii/solution/huan-xing-lian-biao-iiha-xi-biao-kuai-ma-ki2r/
     */
    public ListNode detectCycle4(ListNode head) {
        if (head == null) return null;
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                fast = head;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return slow;
            }
        }
        return null;
    }

    /**
     * K 个一组翻转链表（字节跳动、猿辅导在半年内面试常考）
     * 给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
     * k 是一个正整数，它的值小于或等于链表的长度。
     * 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
     * 进阶：
     * 你可以设计一个只使用常数额外空间的算法来解决此问题吗？
     * 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
     * 示例 1：
     * 输入：head = [1,2,3,4,5], k = 2
     * 输出：[2,1,4,3,5]
     * 示例 2：
     * 输入：head = [1,2,3,4,5], k = 3
     * 输出：[3,2,1,4,5]
     * 示例 3：
     * 输入：head = [1,2,3,4,5], k = 1
     * 输出：[1,2,3,4,5]
     * 示例 4：
     * 输入：head = [1], k = 1
     * 输出：[1]
     * *****************************************************************************************************************
     * 一图胜千言，根据图片看代码，马上就懂了
     * 步骤分解:
     * 1.链表分区为已翻转部分+待翻转部分+未翻转部分
     * 2.每次翻转前，要确定翻转链表的范围，这个必须通过 k 此循环来确定
     * 3.需记录翻转链表前驱和后继，方便翻转完成后把已翻转部分和未翻转部分连接起来
     * 4.初始需要两个变量 pre 和 end，pre 代表待翻转链表的前驱，end 代表待翻转链表的末尾
     * 5.经过k此循环，end 到达末尾，记录待翻转链表的后继 next = end.next
     * 6.翻转链表，然后将三部分链表连接起来，然后重置 pre 和 end 指针，然后进入下一次循环
     * 7.特殊情况，当翻转部分长度不足 k 时，在定位 end 完成后，end==null，已经到达末尾，说明题目已完成，直接返回即可
     * 8.时间复杂度为 O(n*K) 最好的情况为 O(n) 最差的情况未 O(n^2)
     * 9.空间复杂度为 O(1) 除了几个必须的节点指针外，我们并没有占用其他空间
     * 链接：https://leetcode-cn.com/problems/reverse-nodes-in-k-group/solution/tu-jie-kge-yi-zu-fan-zhuan-lian-biao-by-user7208t/
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode pre = dummy;
        ListNode end = dummy;

        while (end.next != null) {
            for (int i = 0; i < k && end != null; i++) end = end.next;
            if (end == null) break;
            ListNode start = pre.next;
            ListNode next = end.next;
            end.next = null;
            pre.next = reverse(start);
            start.next = next;
            pre = start;

            end = pre;
        }
        return dummy.next;
    }

    private ListNode reverse(ListNode head) {
        ListNode pre = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }

    /**
     * 25. Reverse Nodes in k-Group
     * Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
     * k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
     * Follow up:
     * Could you solve the problem in O(1) extra memory space?
     * You may not alter the values in the list's nodes, only nodes itself may be changed.
     * *****************************************************************************************************************
     * Example 1:
     * Input: head = [1,2,3,4,5], k = 2
     * Output: [2,1,4,3,5]
     *
     * Example 2:
     * Input: head = [1,2,3,4,5], k = 3
     * Output: [3,2,1,4,5]
     *
     * Example 3:
     * Input: head = [1,2,3,4,5], k = 1
     * Output: [1,2,3,4,5]
     *
     * Example 4:
     * Input: head = [1], k = 1
     * Output: [1]
     */
    public ListNode reverseKGroup2(ListNode head, int k) {
        int n = 0;
        for (ListNode i = head; i != null; n++, i = i.next) ;

        ListNode dmy = new ListNode(0);
        dmy.next = head;
        for (ListNode prev = dmy, tail = head; n >= k; n -= k) {
            for (int i = 1; i < k; i++) {
                ListNode next = tail.next.next;
                tail.next.next = prev.next;
                prev.next = tail.next;
                tail.next = next;
            }

            prev = tail;
            tail = tail.next;
        }
        return dmy.next;
    }

    /**
     * 1. 两数之和（近半年内，字节跳动在面试中考查此题达到 152 次）
     * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 的那 两个 整数，并返回它们的数组下标。
     * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
     * 你可以按任意顺序返回答案。
     * *****************************************************************************************************************
     * 示例 1：
     * 输入：nums = [2,7,11,15], target = 9
     * 输出：[0,1]
     * 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
     * 示例 2：
     * 输入：nums = [3,2,4], target = 6
     * 输出：[1,2]
     * 示例 3：
     * 输入：nums = [3,3], target = 6
     * 输出：[0,1]
     */
    public int[] twoSum(int[] nums, int target) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] == target) {
                    return new int[]{i, j};
                }
            }
        }
        return new int[0];
    }

    /**
     * 方法二：哈希表
     * 思路及算法
     * 注意到方法一的时间复杂度较高的原因是寻找 target - x 的时间复杂度过高。因此，我们需要一种更优秀的方法，能够快速寻找数组中是否存在目标元素。如果存在，我们需要找出它的索引。
     * 使用哈希表，可以将寻找 target - x 的时间复杂度降低到从 O(N) 降低到 O(1)。
     * 这样我们创建一个哈希表，对于每一个 x，我们首先查询哈希表中是否存在 target - x，然后将 x 插入到哈希表中，即可保证不会让 x 和自己匹配。
     * 链接：https://leetcode-cn.com/problems/two-sum/solution/liang-shu-zhi-he-by-leetcode-solution/
     */
    public int[] twoSum2(int[] nums, int target) {
        Map<Integer, Integer> hashtable = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; ++i) {
            if (hashtable.containsKey(target - nums[i])) {
                return new int[]{hashtable.get(target - nums[i]), i};
            }
            hashtable.put(nums[i], i);
        }
        return new int[0];
    }

    /**
     * 方法3:超哥
     */
    public int[] twoSum3(int[] nums, int target) {
        int[] a = new int[2];
        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) {
                    a[0] = i;
                    a[1] = j;
                    return a;
                }
            }
        }
        return new int[0];
    }

    /**
     * 11. 盛最多水的容器（腾讯、百度、字节跳动在近半年内面试常考）
     * 给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点(i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
     * 说明：你不能倾斜容器。
     * https://leetcode-cn.com/problems/container-with-most-water/solution/container-with-most-water-shuang-zhi-zhen-fa-yi-do/
     */
    public int maxArea(int[] height) {
        int i = 0, j = height.length - 1, res = 0;
        while (i < j) {
            res = height[i] < height[j] ?
                    Math.max(res, (j - i) * height[i++]) :
                    Math.max(res, (j - i) * height[j--]);
        }
        return res;
    }

    /**
     * 双指针法的第二钟解法
     */
    public int maxArea2(int[] height) {
        int max = 0;
        for (int i = 0, j = height.length - 1; i < j; ) {
            int minHeight = height[i] < height[j] ? height[i++] : height[j--];
            // int area = (j-i+1) * minHeight;
            // max = Math.max(max , area);
            max = Math.max(max, (j - i + 1) * minHeight);
        }
        return max;
    }

    /**
     * 用一句话概括双指针解法的要点：指针每一次移动，都意味着排除掉了一个柱子。
     * 如下图所示，在一开始，我们考虑相距最远的两个柱子所能容纳水的面积。水的宽度是两根柱子之间的距离 d = 8 ；水的高度取决于两根柱子之间较短的那个，即左边柱子的高度 h = 3 。水的面积就是 3 * 8 = 24 。
     * 链接：https://leetcode-cn.com/problems/container-with-most-water/solution/on-shuang-zhi-zhen-jie-fa-li-jie-zheng-que-xing-tu/
     * *****************************************************************************************************************
     * 双指针法正确性证明
     * https://leetcode-cn.com/problems/container-with-most-water/solution/shuang-zhi-zhen-fa-zheng-que-xing-zheng-ming-by-r3/
     */
    public int maxArea3(int[] height) {
        int res = 0;
        int i = 0;
        int j = height.length - 1;
        while (i < j) {
            int area = (j - i) * Math.min(height[i], height[j]);
            res = Math.max(res, area);
            if (height[i] < height[j]) {
                i++;
            } else {
                j--;
            }
        }
        return res;
    }

    /**
     * 283. 移动零（华为、字节跳动在近半年内面试常考）
     * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
     * *****************************************************************************************************************
     * 示例:
     *
     * 输入: [0,1,0,3,12]
     * 输出: [1,3,12,0,0]
     */
    public void moveZeroes(int[] nums) {
        int j = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[j] = nums[i];
                if (i != j) {
                    nums[i] = 0;
                }
                System.out.println("nums[" + i + "]=" + nums[i] + "  nums[" + j + "]=" + nums[j]);
                j++;
            }
        }
        // 输出移动零后的数组
        for (int i = 0; i < nums.length; i++) {
            System.out.print("nums[" + i + "] = " + nums[i] + "   ");
        }
        System.out.println();
    }


    public void moveZeroes2(int[] nums) {
        for (int lastNonZeroFoundAt = 0, cur = 0; cur < nums.length; cur++) {
            if (nums[cur] != 0) {
                swap(nums[lastNonZeroFoundAt++], nums[cur]);
            }
        }
    }

    private void swap(int num, int num1) {

    }

    /**
     * 三种方式解决，都击败了100%的用户
     * https://leetcode-cn.com/problems/move-zeroes/solution/san-chong-fang-shi-jie-jue-du-ji-bai-liao-100de-yo/
     * *****************************************************************************************************************
     * 1，把非0的往前挪
     * 把非0的往前挪，挪完之后，后面的就都是0了，然后在用0覆盖后面的。这种是最容易理解也是最容易想到的，代码比较简单，这里就以示例为例画个图来看下
     */
    public void moveZeroes3(int[] nums) {
        if (nums == null || nums.length == 0)
            return;
        int index = 0;
        //一次遍历，把非零的都往前挪
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0)
                nums[index++] = nums[i];
        }
        //后面的都是0,
        while (index < nums.length) {
            nums[index++] = 0;
        }
    }
    /**
     * 2，参照双指针解决
     * 这里可以参照双指针的思路解决，指针j是一直往后移动的，如果指向的值不等于0才对他进行操作。而i统计的是前面0的个数，我们可以把j-i看做另一个指针，它是指向前面第一个0的位置，然后我们让j指向的值和j-i指向的值交换
     */
    public void moveZeroes4(int[] nums) {
        int i = 0;//统计前面0的个数
        for (int j = 0; j < nums.length; j++) {
            if (nums[j] == 0) {//如果当前数字是0就不操作
                i++;
            } else if (i != 0) {
                //否则，把当前数字放到最前面那个0的位置，然后再把
                //当前位置设为0
                nums[j - i] = nums[j];
                nums[j] = 0;
            }
        }
    }

    public void moveZeroes5(int[] nums) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            //只要不为0就往前挪
            if (nums[j] != 0) {
                //i指向的值和j指向的值交换
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
                i++;
            }
        }
    }

    /**
     * Shift non-zero values as far forward as possible
     * Fill remaining space with zeros
     */
    public void moveZeroes6(int[] nums) {
        if (nums == null || nums.length == 0) return;

        int insertPos = 0;
        for (int num : nums) {
            if (num != 0) nums[insertPos++] = num;
        }

        while (insertPos < nums.length) {
            nums[insertPos++] = 0;
        }
    }
    /**
     * THE EASIEST but UNUSUAL snowball JAVA solution BEATS 100% (O(n)) + clear explanation
     * The idea is that we go through the array and gather all zeros on our road.
     * https://leetcode.com/problems/move-zeroes/discuss/172432/THE-EASIEST-but-UNUSUAL-snowball-JAVA-solution-BEATS-100-(O(n))-%2B-clear-explanation
     */
    public void moveZeroes7(int[] nums) {
        int snowBallSize = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                snowBallSize++;
            } else if (snowBallSize > 0) {
                int t = nums[i];
                nums[i] = 0;
                nums[i - snowBallSize] = t;
            }
        }
    }

    /**
     * 70. Climbing Stairs
     * You are climbing a staircase. It takes n steps to reach the top.
     * Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
     * *****************************************************************************************************************
     * Example 1:
     * Input: n = 2
     * Output: 2
     * Explanation: There are two ways to climb to the top.
     * 1. 1 step + 1 step
     * 2. 2 steps
     * Example 2:
     * Input: n = 3
     * Output: 3
     * Explanation: There are three ways to climb to the top.
     * 1. 1 step + 1 step + 1 step
     * 2. 1 step + 2 steps
     * 3. 2 steps + 1 step
     * *****************************************************************************************************************
     * The problem seems to be a dynamic programming one. Hint: the tag also suggests that!
     * Here are the steps to get the solution incrementally.
     *
     * Base cases:
     * if n <= 0, then the number of ways should be zero.
     * if n == 1, then there is only way to climb the stair.
     * if n == 2, then there are two ways to climb the stairs. One solution is one step by another; the other one is two steps at one time.
     *
     * The key intuition to solve the problem is that given a number of stairs n, if we know the number ways to get to the points [n-1] and [n-2] respectively, denoted as n1 and n2 , then the total ways to get to the point [n] is n1 + n2. Because from the [n-1] point, we can take one single step to reach [n]. And from the [n-2] point, we could take two steps to get there.
     *
     * The solutions calculated by the above approach are complete and non-redundant. The two solution sets (n1 and n2) cover all the possible cases on how the final step is taken. And there would be NO overlapping among the final solutions constructed from these two solution sets, because they differ in the final step.
     *
     * Now given the above intuition, one can construct an array where each node stores the solution for each number n. Or if we look at it closer, it is clear that this is basically a fibonacci number, with the starting numbers as 1 and 2, instead of 1 and 1.
     *
     * The implementation in Java as follows:
     */
    public int climbStairs(int n) {
        // base cases
        if (n <= 0) return 0;
        if (n == 1) return 1;
        if (n == 2) return 2;

        int one_step_before = 2;
        int two_steps_before = 1;
        int all_ways = 0;

        for (int i = 2; i < n; i++) {
            all_ways = one_step_before + two_steps_before;
            two_steps_before = one_step_before;
            one_step_before = all_ways;
        }
        return all_ways;
    }

    /**
     * 3-4 short lines in every language
     * Same simple algorithm written in every offered language. Variable a tells you the number of ways to reach the current step, and b tells you the number of ways to reach the next step. So for the situation one step further up, the old b becomes the new a, and the new b is the old a+b, since that new step can be reached by climbing 1 step from what b represented or 2 steps from what a represented.
     *
     * Ruby wins, and "the C languages" all look the same.
     *
     * Ruby (60 ms)
     *
     * def climb_stairs(n)
     *     a = b = 1
     *     n.times { a, b = b, a+b }
     *     a
     * end
     * C++ (0 ms)
     *
     * int climbStairs(int n) {
     *     int a = 1, b = 1;
     *     while (n--)
     *         a = (b += a) - a;
     *     return a;
     * }
     * Java (208 ms)
     *
     * public int climbStairs(int n) {
     *     int a = 1, b = 1;
     *     while (n-- > 0)
     *         a = (b += a) - a;
     *     return a;
     * }
     * Python (52 ms)
     *
     * def climbStairs(self, n):
     *     a = b = 1
     *     for _ in range(n):
     *         a, b = b, a + b
     *     return a
     * C (0 ms)
     *
     * int climbStairs(int n) {
     *     int a = 1, b = 1;
     *     while (n--)
     *         a = (b += a) - a;
     *     return a;
     * }
     * C# (48 ms)
     *
     * public int ClimbStairs(int n) {
     *     int a = 1, b = 1;
     *     while (n-- > 0)
     *         a = (b += a) - a;
     *     return a;
     * }
     * Javascript (116 ms)
     *
     * var climbStairs = function(n) {
     *     a = b = 1
     *     while (n--)
     *         a = (b += a) - a
     *     return a
     * };
     */
    public int climbStairs2(int n) {
        int a = 1, b = 1;
        while (n-- > 0)
            a = (b += a) - a;
        return a;
    }


    public int climbStairs3(int n) {
        if (n <= 2) {
            return n;
        }
        int f1 = 1, f2 = 2, f3 = 3;
        for (int i = 3; i <= n; i++) {
            f3 = f1 + f2;
            f1 = f2;
            f2 = f3;
        }
        return f3;
    }

    /**
     *     动态规划思路： 要考虑第爬到第n阶楼梯时候可能是一步，也可能是两步。
     *     1.计算爬上n-1阶楼梯的方法数量。因为再爬1阶就到第n阶
     *     2.计算爬上n-2阶楼梯体方法数量。因为再爬2阶就到第n阶 那么f(n)=f(n-1)+f(n-2);
     *     为什么不是f(n)=f(n-1)+1+f(n-2)+2呢，因为f(n)是爬楼梯方法数量，不是爬到n阶楼梯的步数
     */
    public int climbStairs4(int n) {
        if (n == 0 || n == 1)
            return n;
        int[] bp = new int[n];
        bp[0] = 1;
        bp[1] = 2;
        for (int i = 2; i < n; i++) {
            bp[i] = bp[i - 1] + bp[i - 2];
        }
        return bp[n - 1];
    }

    // f(n) = f(n-1) + f(n-2);Fibonacci数列
    public int climbStairs5(int n) {
        if (n <= 2) return n;
        return climbStairs(n - 1) + climbStairs(n - 2);
    }

    /**
     *     假设你正在爬楼梯。需要 n 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     *     https://leetcode-cn.com/problems/climbing-stairs/comments/
     *     长度较短的有限集合的解，可直接返回值，自己学习算法最终的目的还是为了更好地解决问题。警醒自己不要沉迷于算法的精妙而忽视实际情况，上了很好的一课(switch case)
     *     第n个台阶只能从第n-1或者n-2个上来。到第n-1个台阶的走法 + 第n-2个台阶的走法 = 到第n个台阶的走法，已经知道了第1个和第2个台阶的走法，一路加上去。
     *     https://leetcode-cn.com/problems/climbing-stairs/solution/pa-lou-ti-by-leetcode-solution/
     *     方法三：通项公式
     */
    public int climbStairs6(int n) {
        double sqrt5 = Math.sqrt(5);
        double fibn = Math.pow((1 + sqrt5) / 2, n + 1) - Math.pow((1 - sqrt5) / 2, n + 1);
        return (int) (fibn / sqrt5);
    }

    /**
     * 15. 三数之和（国内、国际大厂历年面试高频老题）
     * 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
     * 注意：答案中不可以包含重复的三元组。
     * *****************************************************************************************************************
     * Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
     * Notice that the solution set must not contain duplicate triplets.
     * Example 1:
     * Input: nums = [-1,0,1,2,-1,-4]
     * Output: [[-1,-1,2],[-1,0,1]]
     * Example 2:
     * Input: nums = []
     * Output: []
     * Example 3:
     * Input: nums = [0]
     * Output: []
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> lists = new ArrayList<>();
        //排序
        Arrays.sort(nums);
        //双指针
        int len = nums.length;
        for (int i = 0; i < len; ++i) {
            if (nums[i] > 0) return lists;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int curr = nums[i];
            int L = i + 1, R = len - 1;
            while (L < R) {
                int tmp = curr + nums[L] + nums[R];
                if (tmp == 0) {
                    List<Integer> list = new ArrayList<>();
                    list.add(curr);
                    list.add(nums[L]);
                    list.add(nums[R]);
                    lists.add(list);
                    while (L < R && nums[L + 1] == nums[L]) ++L;
                    while (L < R && nums[R - 1] == nums[R]) --R;
                    ++L;
                    --R;
                } else if (tmp < 0) {
                    ++L;
                } else {
                    --R;
                }
            }
        }
        return lists;
    }

    /**
     * 犹豫不决先排序，步步逼近双指针
     * https://leetcode-cn.com/problems/3sum/solution/3sumpai-xu-shuang-zhi-zhen-yi-dong-by-jyd/
     */
    public List<List<Integer>> threeSum2(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int k = 0; k < nums.length - 2; k++) {
            if (nums[k] > 0) break;
            if (k > 0 && nums[k] == nums[k - 1]) continue;
            int i = k + 1, j = nums.length - 1;
            while (i < j) {
                int sum = nums[k] + nums[i] + nums[j];
                if (sum < 0) {
                    while (i < j && nums[i] == nums[++i]) ;
                } else if (sum > 0) {
                    while (i < j && nums[j] == nums[--j]) ;
                } else {
                    res.add(new ArrayList<Integer>(Arrays.asList(nums[k], nums[i], nums[j])));
                    while (i < j && nums[i] == nums[++i]) ;
                    while (i < j && nums[j] == nums[--j]) ;
                }
            }
        }
        return res;
    }

    /**
     * Java with set
     * https://leetcode.com/problems/3sum/discuss/143636/Java-with-set
     * Actually I viewed so many answers but this one is the clearest
     */
    public List<List<Integer>> threeSum3(int[] nums) {
        Set<List<Integer>> res = new HashSet<>();
        if (nums.length == 0) return new ArrayList<>(res);
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            int j = i + 1;
            int k = nums.length - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0) res.add(Arrays.asList(nums[i], nums[j++], nums[k--]));
                else if (sum > 0) k--;
                else if (sum < 0) j++;
            }

        }
        return new ArrayList<>(res);
    }

    /**
     * 第1周 第4课 | 栈、队列、优先队列、双端队列
     * *****************************************************************************************************************
     * 参考链接
     * Java 的 PriorityQueue 文档
     * Java 的 Stack 源码
     * Java 的 Queue 源码
     * Python 的 heapq
     * 高性能的 container 库
     * 预习题目
     * 有效的括号（亚马逊、JPMorgan 在半年内面试常考）
     * 最小栈（亚马逊在半年内面试常考）
     * 实战题目
     * 柱状图中最大的矩形（亚马逊、微软、字节跳动在半年内面试中考过）
     * 滑动窗口最大值（亚马逊在半年内面试常考）
     * 课后作业
     * 用 add first 或 add last 这套新的 API 改写 Deque 的代码
     * 分析 Queue 和 Priority Queue 的源码
     * 设计循环双端队列（Facebook 在 1 年内面试中考过）
     * 接雨水（亚马逊、字节跳动、高盛集团、Facebook 在半年内面试常考）
     * *****************************************************************************************************************
     * 本周作业
     * 简单：
     * 用 add first 或 add last 这套新的 API 改写 Deque 的代码
     * 分析 Queue 和 Priority Queue 的源码
     * 删除排序数组中的重复项（Facebook、字节跳动、微软在半年内面试中考过）
     * 旋转数组（微软、亚马逊、PayPal 在半年内面试中考过）
     * 合并两个有序链表（亚马逊、字节跳动在半年内面试常考）
     * 合并两个有序数组（Facebook 在半年内面试常考）
     * 两数之和（亚马逊、字节跳动、谷歌、Facebook、苹果、微软在半年内面试中高频常考）
     * 移动零（Facebook、亚马逊、苹果在半年内面试中考过）
     * 加一（谷歌、字节跳动、Facebook 在半年内面试中考过）
     * 中等：
     * 设计循环双端队列（Facebook 在 1 年内面试中考过）
     * 困难：
     * 接雨水（亚马逊、字节跳动、高盛集团、Facebook 在半年内面试常考）
     */

    /**
     * 20. 有效的括号（亚马逊、JPMorgan 在半年内面试常考）
     * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
     * 有效字符串需满足：
     * 左括号必须用相同类型的右括号闭合。
     * 左括号必须以正确的顺序闭合。
     * *****************************************************************************************************************
     * 20. Valid Parentheses
     * Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
     * An input string is valid if:
     * Open brackets must be closed by the same type of brackets.
     * Open brackets must be closed in the correct order.
     * *****************************************************************************************************************
     * 【栈】有效的括号
     * 解题思路
     * 根据题意，我们可以推断出以下要点：
     * 1.有效括号字符串的长度，一定是偶数！
     * 2.右括号前面，必须是相对应的左括号，才能抵消！
     * 3.右括号前面，不是对应的左括号，那么该字符串，一定不是有效的括号！
     * 图解演示
     * https://leetcode-cn.com/problems/valid-parentheses/solution/guan-fang-tui-jian-ti-jie-you-xiao-de-gu-zyzg/
     * 示例代码(Golang)
     * func isValid(s string) bool {
     *     n := len(s)
     *     if n % 2 == 1 {
     *         return false
     *     }
     *     pairs := map[byte]byte{
     *         ')': '(',
     *         ']': '[',
     *         '}': '{',
     *     }
     *     stack := []byte{}
     *     for i := 0; i < n; i++ {
     *         if pairs[s[i]] > 0 {
     *             if len(stack) == 0 || stack[len(stack)-1] != pairs[s[i]] {
     *                 return false
     *             }
     *             stack = stack[:len(stack)-1]
     *         } else {
     *             stack = append(stack, s[i])
     *         }
     *     }
     *     return len(stack) == 0
     * }
     *
     * 链接：https://leetcode-cn.com/problems/valid-parentheses/solution/guan-fang-tui-jian-ti-jie-you-xiao-de-gu-zyzg/
     */
    public boolean isValid(String s) {
        int n = s.length();
        if (n % 2 == 1) {
            return false;
        }

        Map<Character, Character> pairs = new HashMap<Character, Character>() {{
            put(')', '(');
            put(']', '[');
            put('}', '{');
        }};
        Deque<Character> stack = new LinkedList<Character>();
        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);
            if (pairs.containsKey(ch)) {
                if (stack.isEmpty() || stack.peek() != pairs.get(ch)) {
                    return false;
                }
                stack.pop();
            } else {
                stack.push(ch);
            }
        }
        return stack.isEmpty();
    }

    /**
     * Short java solution
     * https://leetcode.com/problems/valid-parentheses/discuss/9178/Short-java-solution
     *
     */
    public boolean isValid2(String s) {
        Stack<Character> stack = new Stack<Character>();
        for (char c : s.toCharArray()) {
            if (c == '(')
                stack.push(')');
            else if (c == '{')
                stack.push('}');
            else if (c == '[')
                stack.push(']');
            else if (stack.isEmpty() || stack.pop() != c)
                return false;
        }
        return stack.isEmpty();
    }

    /**
     * 155. 最小栈（亚马逊在半年内面试常考）
     * 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
     * push(x) —— 将元素 x 推入栈中。
     * pop() —— 删除栈顶的元素。
     * top() —— 获取栈顶元素。
     * getMin() —— 检索栈中的最小元素。
     *
     * 示例:
     * 输入：
     * ["MinStack","push","push","push","getMin","pop","top","getMin"]
     * [[],[-2],[0],[-3],[],[],[],[]]
     * 输出：
     * [null,null,null,null,-3,null,0,-2]
     * 解释：
     * MinStack minStack = new MinStack();
     * minStack.push(-2);
     * minStack.push(0);
     * minStack.push(-3);
     * minStack.getMin();   --> 返回 -3.
     * minStack.pop();
     * minStack.top();      --> 返回 0.
     * minStack.getMin();   --> 返回 -2.
     * 提示：
     * pop、top 和 getMin 操作总是在 非空栈 上调用。
     * 见 MinStack.java
     * *****************************************************************************************************************
     */

    /**
     * 84. 柱状图中最大的矩形（亚马逊、微软、字节跳动在半年内面试中考过）
     * 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
     * 求在该柱状图中，能够勾勒出来的矩形的最大面积。
     *
     */
    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        // 特判
        if (len == 0) {
            return 0;
        }

        int res = 0;
        for (int i = 0; i < len; i++) {

            // 找左边最后 1 个大于等于 heights[i] 的下标
            int left = i;
            int curHeight = heights[i];
            while (left > 0 && heights[left - 1] >= curHeight) {
                left--;
            }

            // 找右边最后 1 个大于等于 heights[i] 的索引
            int right = i;
            while (right < len - 1 && heights[right + 1] >= curHeight) {
                right++;
            }

            int width = right - left + 1;
            res = Math.max(res, width * curHeight);
        }
        return res;
    }

    /**
     * 84. Largest Rectangle in Histogram
     * Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.
     * Example 1:
     * Input: heights = [2,1,5,6,2,3]
     * Output: 10
     * Explanation: The above is a histogram where width of each bar is 1.
     * The largest rectangle is shown in the red area, which has an area = 10 units.
     * *****************************************************************************************************************
     * Short and Clean O(n) stack based JAVA solution
     * https://leetcode.com/problems/largest-rectangle-in-histogram/discuss/28900/Short-and-Clean-O(n)-stack-based-JAVA-solution
     */
    public int largestRectangleArea2(int[] heights) {
        int len = heights.length;
        Stack<Integer> s = new Stack<>();
        int maxArea = 0;
        for (int i = 0; i <= len; i++) {
            int h = (i == len ? 0 : heights[i]);
            if (s.isEmpty() || h >= heights[s.peek()]) {
                s.push(i);
            } else {
                int tp = s.pop();
                maxArea = Math.max(maxArea, heights[tp] * (s.isEmpty() ? i : i - 1 - s.peek()));
                i--;
            }
        }
        return maxArea;
    }

    /**
     * Java 双解法 代码简洁易懂 单调栈 + 双指针
     * 视频不知道为什么看不来了， 大家请移步 b站 https://www.bilibili.com/video/BV1J64y1F7Br
     * *****************************************************************************************************************
     * 双指针
     * 可以通过，但是需要调整方法
     */
    public int largestRectangleArea3(int[] heights) {
        if (heights.length == 0) return 0;
        int res = 0;
        int len = heights.length;
        int[] arr = new int[len + 2];
        for (int i = 0; i < len; i++)
            arr[i + 1] = heights[i];

        for (int i = 1; i < arr.length - 1; i++) {
            int left = i, right = i;
            while (left >= 0 && arr[left] >= arr[i])
                left--;
            while (right < arr.length && arr[right] >= arr[i])
                right++;

            res = Math.max((right - left - 1) * arr[i], res);
        }

        return res;
    }

    /**
     * 单调栈解法
     */
    public int largestRectangleArea4(int[] heights) {
        if (heights.length == 0) return 0;
        int res = 0;
        int len = heights.length;
        int[] arr = new int[len + 2];
        for (int i = 0; i < len; i++)
            arr[i + 1] = heights[i];

        Deque<Integer> stack = new ArrayDeque<>();
        int index = 1;
        stack.push(0);
        while (index < arr.length) {
            while (index < arr.length && arr[index] >= arr[stack.peek()])
                stack.push(index++);

            while (index < arr.length && arr[index] < arr[stack.peek()]) {
                int curHeight = arr[stack.pop()];
                res = Math.max(res, curHeight * (index - stack.peek() - 1));
            }

            stack.push(index);
        }

        return res;
    }

    /**
     * 239. 滑动窗口最大值（亚马逊在半年内面试中常考）
     * 给定一个数组 nums，有一个大小为k的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k个数字。滑动窗口每次只向右移动一位。
     * 返回滑动窗口中的最大值。
     * *****************************************************************************************************************
     * 示例 1：
     * 输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
     * 输出: [3,3,5,5,6,7]
     * 解释:
     *
     *   滑动窗口的位置                最大值
     * ---------------               -----
     * [1  3  -1] -3  5  3  6  7       3
     *  1 [3  -1  -3] 5  3  6  7       3
     *  1  3 [-1  -3  5] 3  6  7       5
     *  1  3  -1 [-3  5  3] 6  7       5
     *  1  3  -1  -3 [5  3  6] 7       6
     *  1  3  -1  -3  5 [3  6  7]      7
     * *****************************************************************************************************************
     * 方法一：暴力法
     * 直觉
     * 最简单直接的方法是遍历每个滑动窗口，找到每个窗口的最大值。一共有 N - k + 1 个滑动窗口，每个有 k 个元素，于是算法的时间复杂度为 O(Nk)，表现较差。
     * 链接：https://leetcode-cn.com/problems/sliding-window-maximum/solution/hua-dong-chuang-kou-zui-da-zhi-by-leetcode-3/
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        if (n * k == 0) return new int[0];

        int[] output = new int[n - k + 1];
        for (int i = 0; i < n - k + 1; i++) {
            int max = Integer.MIN_VALUE;
            for (int j = i; j < i + k; j++)
                max = Math.max(max, nums[j]);
            output[i] = max;
        }
        return output;
    }

    /**
     * 双向队列解决滑动窗口最大值
     * 此方法是看了左程云算法课讲解的思路，我只是知识的搬运工。当时听了之后，自己来做了一遍，并发表了评论。没想到现在成了热评，而且现在 LeetCode 可以自己写题解，官方既然发话了，我就再重新整理一下。
     * 思路
     * 遍历数组，将 数 存放在双向队列中，并用 L,R 来标记窗口的左边界和右边界。队列中保存的并不是真的 数，而是该数值对应的数组下标位置，并且数组中的数要从大到小排序。如果当前遍历的数比队尾的值大，则需要弹出队尾值，直到队列重新满足从大到小的要求。刚开始遍历时，L 和 R 都为 0，有一个形成窗口的过程，此过程没有最大值，L 不动，R 向右移。当窗口大小形成时，L 和 R 一起向右移，每次移动时，判断队首的值的数组下标是否在 [L,R] 中，如果不在则需要弹出队首的值，当前窗口的最大值即为队首的数。
     * 示例
     * 输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
     * 输出: [3,3,5,5,6,7]
     * 解释过程中队列中都是具体的值，方便理解，具体见代码。
     * 初始状态：L=R=0,队列:{}
     * i=0,nums[0]=1。队列为空,直接加入。队列：{1}
     * i=1,nums[1]=3。队尾值为1，3>1，弹出队尾值，加入3。队列：{3}
     * i=2,nums[2]=-1。队尾值为3，-1<3，直接加入。队列：{3,-1}。此时窗口已经形成，L=0,R=2，result=[3]
     * i=3,nums[3]=-3。队尾值为-1，-3<-1，直接加入。队列：{3,-1,-3}。队首3对应的下标为1，L=1,R=3，有效。result=[3,3]
     * i=4,nums[4]=5。队尾值为-3，5>-3，依次弹出后加入。队列：{5}。此时L=2,R=4，有效。result=[3,3,5]
     * i=5,nums[5]=3。队尾值为5，3<5，直接加入。队列：{5,3}。此时L=3,R=5，有效。result=[3,3,5,5]
     * i=6,nums[6]=6。队尾值为3，6>3，依次弹出后加入。队列：{6}。此时L=4,R=6，有效。result=[3,3,5,5,6]
     * i=7,nums[7]=7。队尾值为6，7>6，弹出队尾值后加入。队列：{7}。此时L=5,R=7，有效。result=[3,3,5,5,6,7]
     * 通过示例发现 R=i，L=k-R。由于队列中的值是从大到小排序的，所以每次窗口变动时，只需要判断队首的值是否还在窗口中就行了。
     * 解释一下为什么队列中要存放数组下标的值而不是直接存储数值，因为要判断队首的值是否在窗口范围内，由数组下标取值很方便，而由值取数组下标不是很方便。
     * 代码
     * 作者：hanyuhuang
     * 链接：https://leetcode-cn.com/problems/sliding-window-maximum/solution/shuang-xiang-dui-lie-jie-jue-hua-dong-chuang-kou-2/
     */
    public int[] maxSlidingWindow2(int[] nums, int k) {
        if (nums == null || nums.length < 2) return nums;
        // 双向队列 保存当前窗口最大值的数组位置 保证队列中数组位置的数值按从大到小排序
        LinkedList<Integer> queue = new LinkedList();
        // 结果数组
        int[] result = new int[nums.length - k + 1];
        // 遍历nums数组
        for (int i = 0; i < nums.length; i++) {
            // 保证从大到小 如果前面数小则需要依次弹出，直至满足要求
            while (!queue.isEmpty() && nums[queue.peekLast()] <= nums[i]) {
                queue.pollLast();
            }
            // 添加当前值对应的数组下标
            queue.addLast(i);
            // 判断当前队列中队首的值是否有效
            if (queue.peek() <= i - k) {
                queue.poll();
            }
            // 当窗口长度为k时 保存当前窗口中最大值
            if (i + 1 >= k) {
                result[i + 1 - k] = nums[queue.peek()];
            }
        }
        return result;
    }

    /**
     * 单调队列也可以存值，题解中存的是下标。
     * 如果存值的话，每次只有新元素 大于 队列尾部的元素时，才去移除队列尾部的元素
     * 窗口左侧移出去的元素如果等于队列头部的元素，则removeFirst。
     * 举个例子： "543321" ，k=3
     * 队列存值的情况下，如果不将两个3都加入，当第一个3被移出时，会导致321的最大值变为2，因为3已经被移出了，因此存值的话，需要新的元素大于队列尾部元素再去移除队列尾部的元素。
     * 队列存下标的情况下，就可以只存一个3（存第二个），因为通过下标就能判断出移出的是第一个3还是第二个3。
     * https://leetcode-cn.com/problems/sliding-window-maximum/solution/shuang-xiang-dui-lie-jie-jue-hua-dong-chuang-kou-2/
     */
    public int[] maxSlidingWindow3(int[] nums, int k) {
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> deque = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            while (!deque.isEmpty() && deque.peekLast() < nums[i]) {//注意：此处是小于号
                deque.removeLast();
            }
            deque.addLast(nums[i]);
            if (i >= k && nums[i - k] == deque.peekFirst()) {
                deque.removeFirst();
            }
            if (i >= k - 1) {
                res[i - k + 1] = deque.peekFirst();
            }
        }
        return res;
    }

    /**
     * 优先队列（最大堆）、 单调队列，总有一款适合你
     * 题目描述：
     * 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。返回滑动窗口中的最大值。
     * 方法一：最大堆(优先队列)
     * 我们可以采用最大堆的数据结构来保存元素，堆顶元素即为当前堆的最大值，并判断当前堆顶元素这是否在窗口中，在则直接返回，不在则删除堆顶元素并调整堆。
     * 我们以nums = [1,3,-1,-3,5,3,6,7], k = 3为例来模拟堆的过程。如下图所示：
     * 链接：https://leetcode-cn.com/problems/sliding-window-maximum/solution/you-xian-dui-lie-zui-da-dui-dan-diao-dui-dbn9/
     * 上述如何创建堆和调整堆暂不讨论,感兴趣可自行百度。在Java中的优先队列(PriorityQueue)就是堆的数据结构。代码如下：
     *
     * 方法一：最大堆(优先队列)
     */
    public int[] maxSlidingWindow4(int[] nums, int k) {
        int n = nums.length;
        //这里我们传入了一个比较器，当两者的值相同时，比较下标的位置，下标大的在前面。
        PriorityQueue<int[]> queue = new PriorityQueue<>((p1, p2) -> p1[0] != p2[0] ? p2[0] - p1[0] : p2[1] - p1[1]);
        //初始化前K的元素到堆中
        for (int i = 0; i < k; i++) {
            queue.offer(new int[]{nums[i], i});
        }
        //有n-k+1个
        int[] ans = new int[n - k + 1];
        //将第一次答案加入数据
        ans[0] = queue.peek()[0];
        for (int i = k; i < n; i++) {
            //将新元素加入优先队列
            queue.offer(new int[]{nums[i], i});
            //循环判断当前队首是否在窗口中，窗口的左边界为i-k
            while (queue.peek()[1] <= i - k) {
                queue.poll();
            }
            //在窗口中直接赋值即可
            ans[i - k + 1] = queue.peek()[0];
        }
        return ans;
    }

    /**
     * 方法二：单调队列(双端队列)
     * 首先，明确一个概念，单调队列是在队列中所有的元素都是单调的，要么单调增，要么单调减，新增元素时到队尾，此时队列不符合单调性就将队尾元素出队，直到队列单调或空。
     * 还是以nums = [1,3,-1,-3,5,3,6,7], k = 3为例来模拟对列的过程。(队列中的元素是单调递减的)
     * 链接：https://leetcode-cn.com/problems/sliding-window-maximum/solution/you-xian-dui-lie-zui-da-dui-dan-diao-dui-dbn9/
     * 代码如下：
     */
    public int[] maxSlidingWindow5(int[] nums, int k) {
        int n = nums.length;
        //创建双端队列
        Deque<Integer> deque = new ArrayDeque<>();
        //先初始化前K个元素
        for (int i = 0; i < k; i++) {
            //判断队列是否为空 或者当前入队元素是否大于队尾元素 大于则出队
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            //当前元素入队
            //由于需要判断当前元素是否在窗口中，所以实际上队列中存储的为当前元素的下标
            //根据下标找元素比根据元素找下标方便
            deque.offerLast(i);
        }
        int[] ans = new int[n - k + 1];
        //添加当前最大元素
        ans[0] = nums[deque.peekFirst()];
        for (int i = k; i < n; i++) {
            //判断队列是否为空 或者当前入队元素是否大于队尾元素 大于则出队
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            //当前元素入队
            deque.offerLast(i);
            //判断队首元素是否在窗口中
            while (deque.peekFirst() <= i - k) {
                deque.pollFirst();
            }
            //添加答案
            ans[i - k + 1] = nums[deque.peekFirst()];
        }
        return ans;
    }

    /**
     * 【动画模拟】一下就能读懂（单调双端队列
     * 这个题目，算是很经典的题目，我们的滑动窗口主要分为两种，一种的可变长度的滑动窗口，一种是固定长度的滑动窗口，这个题目算是固定长度的代表。今天我们用双端队列来解决我们这个题目，学会了这个题目的解题思想你可以去解决一下两道题目 剑指 Offer 59 - II. 队列的最大值，155. 最小栈，虽然这两个题目和该题类型不同，但是解题思路是一致的，都是很不错的题目，我认为做题，那些考察的很细的，解题思路很难想，即使想到，也不容易完全写出来的题目，才是能够大大提高我们编码能力的题目，希望能和大家一起进步。
     * 这个题目我们用到了双端队列，队列里面保存的则为每段滑动窗口的最大值，我给大家做了一个动图，先来看一下代码执行过程吧。
     * 我们先来了解下双端队列吧，队列我们都知道，是先进先出，双端队列呢？既可以从队头出队，也可以从队尾出队，则不用遵循先进先出的规则。
     * 下面我们通过一个动画来了解一下吧。
     * 链接：https://leetcode-cn.com/problems/sliding-window-maximum/solution/zhe-hui-yi-miao-dong-bu-liao-liao-de-hua-7fy5/
     * 好啦，我们了解双端队列是什么东东了，下面我们通过一个动画，来看一下代码的执行过程吧，相信各位一下就能够理解啦。
     * 我们就通过题目中的例子来表述。nums = [1,3,-1,-3,5,3,6,7], k = 3
     * 链接：https://leetcode-cn.com/problems/sliding-window-maximum/solution/zhe-hui-yi-miao-dong-bu-liao-liao-de-hua-7fy5/
     * 我们将执行过程进行拆解。
     * 1.想将我们第一个窗口的所有值存入单调双端队列中，单调队列里面的值为单调递减的。如果发现队尾元素小于要加入的元素，则将队尾元素出队，直到队尾元素大于新元素时，再让新元素入队，目的就是维护一个单调递减的队列。
     * 2.我们将第一个窗口的所有值，按照单调队列的规则入队之后，因为队列为单调递减，所以队头元素必为当前窗口的最大值，则将队头元素添加到数组中。
     * 3.移动窗口，判断当前窗口前的元素是否和队头元素相等，如果相等则出队。
     * 4.继续然后按照规则进行入队，维护单调递减队列。
     * 5.每次将队头元素存到返回数组里。
     * 6.返回数组
     * 是不是懂啦，再回去看一遍视频吧。祝大家新年快乐，天天开心呀！
     * 链接：https://leetcode-cn.com/problems/sliding-window-maximum/solution/zhe-hui-yi-miao-dong-bu-liao-liao-de-hua-7fy5/
     */
    public int[] maxSlidingWindow6(int[] nums, int k) {
        int len = nums.length;
        if (len == 0) {
            return nums;
        }
        int[] arr = new int[len - k + 1];
        int arr_index = 0;
        //我们需要维护一个单调递增的双向队列
        Deque<Integer> deque = new LinkedList<>();
        //先将第一个窗口的值按照规则入队
        for (int i = 0; i < k; i++) {
            while (!deque.isEmpty() && deque.peekLast() < nums[i]) {
                deque.removeLast();
            }
            deque.offerLast(nums[i]);
        }
        //存到数组里，队头元素
        arr[arr_index++] = deque.peekFirst();
        //移动窗口
        for (int j = k; j < len; j++) {
            //对应咱们的红色情况，则是窗口的前一个元素等于队头元素
            if (nums[j - k] == deque.peekFirst()) {
                deque.removeFirst();
            }
            while (!deque.isEmpty() && deque.peekLast() < nums[j]) {
                deque.removeLast();
            }
            deque.offerLast(nums[j]);
            arr[arr_index++] = deque.peekFirst();
        }
        return arr;
    }

    /**
     * 中等：
     * 设计循环双端队列（Facebook 在 1 年内面试中考过）见 MyCircularDeque.java
     * *****************************************************************************************************************
     * 641. Design Circular Deque
     * Design your implementation of the circular double-ended queue (deque).
     *
     * Your implementation should support following operations:
     *
     * MyCircularDeque(k): Constructor, set the size of the deque to be k.
     * insertFront(): Adds an item at the front of Deque. Return true if the operation is successful.
     * insertLast(): Adds an item at the rear of Deque. Return true if the operation is successful.
     * deleteFront(): Deletes an item from the front of Deque. Return true if the operation is successful.
     * deleteLast(): Deletes an item from the rear of Deque. Return true if the operation is successful.
     * getFront(): Gets the front item from the Deque. If the deque is empty, return -1.
     * getRear(): Gets the last item from Deque. If the deque is empty, return -1.
     * isEmpty(): Checks whether Deque is empty or not.
     * isFull(): Checks whether Deque is full or not.
     *
     *
     * Example:
     *
     * MyCircularDeque circularDeque = new MycircularDeque(3); // set the size to be 3
     * circularDeque.insertLast(1);			// return true
     * circularDeque.insertLast(2);			// return true
     * circularDeque.insertFront(3);			// return true
     * circularDeque.insertFront(4);			// return false, the queue is full
     * circularDeque.getRear();  			// return 2
     * circularDeque.isFull();				// return true
     * circularDeque.deleteLast();			// return true
     * circularDeque.insertFront(4);			// return true
     * circularDeque.getFront();			// return 4
     * *****************************************************************************************************************
     * https://leetcode.com/problems/design-circular-deque/
     */

    /**
     * 42. Trapping Rain Water
     * Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
     * *****************************************************************************************************************
     * Example 1:
     *
     * Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
     * Output: 6
     * Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
     *
     * Example 2:
     *
     * Input: height = [4,2,0,3,2,5]
     * Output: 9
     * *****************************************************************************************************************
     * https://leetcode.com/problems/trapping-rain-water/
     * *****************************************************************************************************************
     * 困难：
     * 接雨水（亚马逊、字节跳动、高盛集团、Facebook 在半年内面试常考）
     *
     * 详细通俗的思路分析，多解法
     * 思路:
     * 黑色的看成墙，蓝色的看成水，宽度一样，给定一个数组，每个数代表从左到右墙的高度，求出能装多少单位的水。也就是图中蓝色正方形的个数。
     * 解法一：按行求
     * 解法二：按列求
     * 解法三：动态规划
     * 解法四：双指针
     * 解法五：栈
     * https://leetcode-cn.com/problems/trapping-rain-water/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-w-8/
     *
     * 解法一： 月暗送湖风， 相寻路不通。 知在此塘中，AC 不让过。
     * 解法二： 大力出奇迹，两个 for 循环。
     * 解法三： 空间换时间，有房是大爷。
     * 解法四： 四两拨千斤，两个小指针。
     * 解法五： 欲穷世间理，上下而求索。
     * 从解法一看到解法五，我仿佛看到了一个程序员不断通过总结，快速成长的过程。佩服，佩服！
     * https://leetcode-cn.com/problems/trapping-rain-water/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-w-8/
     */
    public int trap(int[] height) {
        int sum = 0;
        int max = getMax(height);//找到最大的高度，以便遍历。
        for (int i = 1; i <= max; i++) {
            boolean isStart = false; //标记是否开始更新 temp
            int temp_sum = 0;
            for (int j = 0; j < height.length; j++) {
                if (isStart && height[j] < i) {
                    temp_sum++;
                }
                if (height[j] >= i) {
                    sum = sum + temp_sum;
                    temp_sum = 0;
                    isStart = true;
                }
            }
        }
        return sum;
    }

    private int getMax(int[] height) {
        int max = 0;
        for (int i = 0; i < height.length; i++) {
            if (height[i] > max) {
                max = height[i];
            }
        }
        return max;
    }

    /**
     * 解法二：按列求
     */
    public int trap2(int[] height) {
        int sum = 0;
        //最两端的列不用考虑，因为一定不会有水。所以下标从 1 到 length - 2
        for (int i = 1; i < height.length - 1; i++) {
            int max_left = 0;
            //找出左边最高
            for (int j = i - 1; j >= 0; j--) {
                if (height[j] > max_left) {
                    max_left = height[j];
                }
            }
            int max_right = 0;
            //找出右边最高
            for (int j = i + 1; j < height.length; j++) {
                if (height[j] > max_right) {
                    max_right = height[j];
                }
            }
            //找出两端较小的
            int min = Math.min(max_left, max_right);
            //只有较小的一段大于当前列的高度才会有水，其他情况不会有水
            if (min > height[i]) {
                sum = sum + (min - height[i]);
            }
        }
        return sum;
    }

    /**
     * 解法三: 动态规划
     */
    public int trap3(int[] height) {
        int sum = 0;
        int[] max_left = new int[height.length];
        int[] max_right = new int[height.length];

        for (int i = 1; i < height.length - 1; i++) {
            max_left[i] = Math.max(max_left[i - 1], height[i - 1]);
        }
        for (int i = height.length - 2; i >= 0; i--) {
            max_right[i] = Math.max(max_right[i + 1], height[i + 1]);
        }
        for (int i = 1; i < height.length - 1; i++) {
            int min = Math.min(max_left[i], max_right[i]);
            if (min > height[i]) {
                sum = sum + (min - height[i]);
            }
        }
        return sum;
    }

    /**
     * 解法四：双指针
     */
    public int trap4(int[] height) {
        int sum = 0;
        int max_left = 0;
        int[] max_right = new int[height.length];
        for (int i = height.length - 2; i >= 0; i--) {
            max_right[i] = Math.max(max_right[i + 1], height[i + 1]);
        }
        for (int i = 1; i < height.length - 1; i++) {
            max_left = Math.max(max_left, height[i - 1]);
            int min = Math.min(max_left, max_right[i]);
            if (min > height[i]) {
                sum = sum + (min - height[i]);
            }
        }
        return sum;
    }

    /**
     * 解法四：双指针
     * *****************************************************************************************************************
     * 我觉得你讲双指针那里完全没讲清楚，官方题解下的这个评论倒是讲得很好。
     * https://leetcode-cn.com/problems/trapping-rain-water/solution/jie-yu-shui-by-leetcode/327718/
     */
    public int trap5(int[] height) {
        int sum = 0;
        int max_left = 0;
        int max_right = 0;
        int left = 1;
        int right = height.length - 2; // 加右指针进去
        for (int i = 1; i < height.length - 1; i++) {
            //从左到右更
            if (height[left - 1] < height[right + 1]) {
                max_left = Math.max(max_left, height[left - 1]);
                int min = max_left;
                if (min > height[left]) {
                    sum = sum + (min - height[left]);
                }
                left++;
                //从右到左更
            } else {
                max_right = Math.max(max_right, height[right + 1]);
                int min = max_right;
                if (min > height[right]) {
                    sum = sum + (min - height[right]);
                }
                right--;
            }
        }
        return sum;
    }

    /**
     * 解法四：双指针
     * *****************************************************************************************************************
     * 纯C的双指针：
     * 见 https://leetcode-cn.com/problems/trapping-rain-water/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-w-8/ 里的评论
     */
    public int trap7(int[] height) {
        int capacity;   /* 雨水容量 */
        int i, j;       /* 双指针 */
        int left_max, right_max;

        if (height == null || height.length < 3) {
            return 0;
        }
        i = 0;
        j = height.length - 1;
        left_max = height[i];
        right_max = height[j];
        capacity = 0;

        while (i < j) {
            if (left_max < right_max) {
                while (height[i] <= left_max && i < j) {
                    capacity += left_max - height[i];
                    ++i;
                }
                left_max = height[i];
            } else {
                while (height[j] <= right_max && i < j) {
                    capacity += right_max - height[j];
                    --j;
                }
                right_max = height[j];
            }
        }
        return capacity;
    }

    /**
     * 解法四：双指针
     * *****************************************************************************************************************
     *             使用双指针（左右两边各两个指针）
     *
     *             我们使用一根一根柱子计算装水量的方法
     *
     *             left 表示左边当前遍历的柱子（即左边我们需要计算能够装多少水的柱子）
     *             left_max 表示 left 的左边最高的柱子长度（不包括 left）
     *             right 表示右边当前遍历的柱子
     *             right_max 表示 right 的右边最高的柱子长度（不包括 right）
     *
     *             我们有以下几个公式：
     *             当 left_max < right_max 的话，那么我们就判断 left_max 是否比 left 高
     *                 因为根据木桶效应，一个桶装水量取决于最短的那个木板，这里也一样，柱子能否装水取决于左右两边的是否都存在比它高的柱子
     *                 因为 left_max < right_max 了，那么我们只需要比较 left_max 即可
     *                     如果 left_max > left，那么装水量就是 left_max - left
     *                     如果 left_max <= left，那么装水量为 0，即 left 装不了水
     *             当 left_max >= right_max 的话，同理如上，比较 right_max 和 right
     *
     *             ？？？？ 为什么 right_max 和 left 隔这么远我们还可以使用 right_max 来判断？
     *             前提：left_max < right_max
     *             right_max 虽然跟 left 离得远，但有如下两种情况：
     *             1、left 柱子和 right_max 柱子之间，没有比 right_max 柱子更高的柱子了，
     *             那么情况如下：  left 能否装水取决于 left_max 柱子是否比 left 高
     *                             |
     *                 |           |
     *                 |   |       |
     *                 ↑   ↑       ↑
     *                l_m  l      r_m
     *
     *             2、left 柱子和 right_max 柱子之间存在比 right_max 柱子更高的柱子
     *             那么情况如下：因为存在了比 right_max 更高的柱子，那么我们仍然只需要判断 left_max 是否比 left 高，因为右边已经存在比 left 高的柱子
     *                         |
     *                         |   |
     *                 |       |   |
     *                 |   |   |   |
     *                 ↑   ↑   ↑   ↑
     *                l_m  l  mid  r_m
     *
     *             初始化指针：
     *             left = 1;
     *             right = len - 2;
     *             left_max = 0;
     *             right_max = len - 1;
     *             （因为第一个柱子和最后一个柱子肯定不能装水，因为不作为装水柱子，而是作为左边最高柱子和右边最高柱子）
     *
     * 见 https://leetcode-cn.com/problems/trapping-rain-water/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-w-8/ 里的评论
     */
    public int trap8(int[] height) {
        int len = height.length;
        int left = 1;
        int right = len - 2;
        int left_max = 0;
        int right_max = len - 1;
        int res = 0;
        while (left <= right) {
            //比较
            if (height[left_max] < height[right_max]) {
                if (height[left_max] > height[left]) {
                    res += height[left_max] - height[left];
                } else {
                    left_max = left;
                }
                left++;
            } else {
                if (height[right_max] > height[right]) {
                    res += height[right_max] - height[right];
                } else {
                    right_max = right;
                }
                right--;
            }
        }
        return res;
    }

    /**
     * 解法五：栈
     * *****************************************************************************************************************
     * 栈求distance peek看的不是index，没法求出来distance，需要一个indexof
     */
    public int trap6(int[] height) {
        int sum = 0;
        Stack<Integer> stack = new Stack<>();
        int current = 0;
        while (current < height.length) {
            //如果栈不空并且当前指向的高度大于栈顶高度就一直循环
            while (!stack.empty() && height[current] > height[stack.peek()]) {
                int h = height[stack.peek()]; //取出要出栈的元素
                stack.pop(); //出栈
                if (stack.empty()) { // 栈空就出去
                    break;
                }
                int distance = current - stack.peek() - 1; //两堵墙之前的距离。
                int min = Math.min(height[stack.peek()], height[current]);
                sum = sum + distance * (min - h);
            }
            stack.push(current); //当前指向的墙入栈
            current++; //指针后移
        }
        return sum;
    }

    /**
     * 26. 删除有序数组中的重复项（Facebook、字节跳动、微软在半年内面试中考过）
     * 给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。
     * 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
     * 说明:
     * 为什么返回数值是整数，但输出的答案是数组呢?
     * 请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。
     * 你可以想象内部操作如下:
     * // nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
     * int len = removeDuplicates(nums);
     *
     * // 在函数里修改输入数组对于调用者是可见的。
     * // 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
     * for (int i = 0; i < len; i++) {
     *  print(nums[i]);
     * }
     * *****************************************************************************************************************
     * 示例 1：
     *
     * 输入：nums = [1,1,2]
     * 输出：2, nums = [1,2]
     * 解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
     * 示例 2：
     *
     * 输入：nums = [0,0,1,1,1,2,2,3,3,4]
     * 输出：5, nums = [0,1,2,3,4]
     * 解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。
     *
     * 链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array
     * *****************************************************************************************************************
     * 【双指针】删除重复项-带优化思路
     * https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/solution/shuang-zhi-zhen-shan-chu-zhong-fu-xiang-dai-you-hu/
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int p = 0;
        int q = 1;
        while (q < nums.length) {
            if (nums[p] != nums[q]) {
                nums[p + 1] = nums[q];
                p++;
            }
            q++;
        }
        return p + 1;
    }

    public int removeDuplicates2(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int p = 0;
        int q = 1;
        while (q < nums.length) {
            if (nums[p] != nums[q]) {
                if (q - p > 1) {
                    nums[p + 1] = nums[q];
                }
                p++;
            }
            q++;
        }
        return p + 1;
    }

    /**
     * 189. 旋转数组（微软、亚马逊、PayPal 在半年内面试中考过）
     * 给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。
     * 尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
     * 你可以使用空间复杂度为 O(1) 的 原地算法解决这个问题吗？
     * 示例 1:
     * 输入: nums = [1,2,3,4,5,6,7], k = 3
     * 输出: [5,6,7,1,2,3,4]
     * 解释:
     * 向右旋转 1 步: [7,1,2,3,4,5,6]
     * 向右旋转 2 步: [6,7,1,2,3,4,5]
     * 向右旋转 3 步: [5,6,7,1,2,3,4]
     * 示例 2:
     *
     * 输入：nums = [-1,-100,3,99], k = 2
     * 输出：[3,99,-1,-100]
     * 解释:
     * 向右旋转 1 步: [99,-1,-100,3]
     * 向右旋转 2 步: [3,99,-1,-100]
     * *****************************************************************************************************************
     * 【数组翻转】旋转数组
     * 解题思路
     * 根据题意，如果使用多余数组存储空间，会导致空间复杂度为 n，所以在这里，我们可以使用常量级的空间复杂度解法：数组翻转。
     * 思路如下：
     * 1.首先对整个数组实行翻转，这样子原数组中需要翻转的子数组，就会跑到数组最前面。
     * 2.这时候，从 k 处分隔数组，左右两数组，各自进行翻转即可。
     * 图解演示
     * https://leetcode-cn.com/problems/rotate-array/solution/shu-zu-fan-zhuan-xuan-zhuan-shu-zu-by-de-5937/
     * *****************************************************************************************************************
     * 2，多次反转
     * 先反转全部数组，在反转前k个，最后在反转剩余的，如下所示
     * 链接：https://leetcode-cn.com/problems/rotate-array/solution/javadai-ma-3chong-fang-shi-tu-wen-xiang-q8lz9/
     *
     */
    public void rotate(int[] nums, int k) {
        int length = nums.length;
        k %= length;
        reverse(nums, 0, length - 1);//先反转全部的元素
        reverse(nums, 0, k - 1);//再反转前k个元素
        reverse(nums, k, length - 1);//接着反转剩余的
    }

    //把数组中从[start，end]之间的元素两两交换,也就是反转
    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start++] = nums[end];
            nums[end--] = temp;
        }
    }

    /**
     * 其实还可以在调整下，先反转前面的，接着反转后面的k个，最后在反转全部，原理都一样
     */
    public void rotate2(int[] nums, int k) {
        int length = nums.length;
        k %= length;
        reverse2(nums, 0, length - k - 1);//先反转前面的
        reverse2(nums, length - k, length - 1);//接着反转后面k个
        reverse2(nums, 0, length - 1);//最后在反转全部的元素
    }

    //把数组中从[start，end]之间的元素两两交换,也就是反转
    public void reverse2(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start++] = nums[end];
            nums[end--] = temp;
        }
    }

    /**
     * 3，环形旋转
     * 类似约瑟夫环一样，把数组看作是环形的，每一个都往后移动k位，这个很好理解
     * https://leetcode-cn.com/problems/rotate-array/solution/javadai-ma-3chong-fang-shi-tu-wen-xiang-q8lz9/
     */
    public void rotate3(int[] nums, int k) {
        int hold = nums[0];
        int index = 0;
        int length = nums.length;
        boolean[] visited = new boolean[length];
        for (int i = 0; i < length; i++) {
            index = (index + k) % length;
            if (visited[index]) {
                //如果访问过，再次访问的话，会出现原地打转的现象，
                //不能再访问当前元素了，我们直接从他的下一个元素开始
                index = (index + 1) % length;
                hold = nums[index];
                i--;
            } else {
                //把当前值保存在下一个位置，保存之前要把下一个位置的
                //值给记录下来
                visited[index] = true;
                int temp = nums[index];
                nums[index] = hold;
                hold = temp;
            }
        }
    }

    /**
     * 21. 合并两个有序链表（亚马逊、字节跳动在半年内面试常考）
     * 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
     * *****************************************************************************************************************
     * 示例 1：
     *
     * 输入：l1 = [1,2,4], l2 = [1,3,4]
     * 输出：[1,1,2,3,4,4]
     * 示例 2：
     *
     * 输入：l1 = [], l2 = []
     * 输出：[]
     * 示例 3：
     *
     * 输入：l1 = [], l2 = [0]
     * 输出：[0]
     * *****************************************************************************************************************
     * 【递归】合并两个有序链表
     * 🧠 解题思路
     * 根据题意，我们知道我们的任务是要将两个升序链表合并为一个升序的新链表！
     * 这就好比，军训的时候，有两个小组，每一组都是按照身高，从左往右依次站立的。
     * 这时候，教官让我们两个小组，合并为一个小组，并且也要按照身高来站立。
     * 这个时候，是不是感觉题目，有一点温度了，人间，又充满阳光了？
     * 跟着你的感觉，我们来想象一下如何正确的合并成一个小组，流程如下：
     * 首先，我们给小组命名，一组为 A，一组为 B，新组合的的 C 组。
     * 对比 A 组和 B 组现在站在最前面的人的身高，矮的先出来，站在 C 组第一位。
     * 然后再次对比两组开头的人的身高，矮的又站出来，站在 C 组第二位。
     * 就这样周而复始，最终，AB 两组的人，全部站到了 C 组，我们的任务也就完成了。
     * 而我们实现该逻辑的方法，就是：递归！
     * 🎨 图解演示
     * 作者：demigodliu
     * 链接：https://leetcode-cn.com/problems/merge-two-sorted-lists/solution/di-gui-he-bing-liang-ge-you-xu-lian-biao-hghk/
     * 🍭 示例代码
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        } else if (l2 == null) {
            return l1;
        } else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }




    /**
     * 88. 合并两个有序数组（Facebook 在半年内面试常考
     * 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
     * 初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。
     * *****************************************************************************************************************
     * 示例 1：
     *
     * 输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
     * 输出：[1,2,2,3,5,6]
     * 示例 2：
     *
     * 输入：nums1 = [1], m = 1, nums2 = [], n = 0
     * 输出：[1]
     * *****************************************************************************************************************
     * 解题方案
     * 思路
     * 标签：从后向前数组遍历
     * 因为 nums1 的空间都集中在后面，所以从后向前处理排序的数据会更好，节省空间，一边遍历一边将值填充进去
     * 设置指针 len1 和 len2 分别指向 nums1 和 nums2 的有数字尾部，从尾部值开始比较遍历，同时设置指针 len 指向 nums1 的最末尾，每次遍历比较值大小之后，则进行填充
     * 当 len1<0 时遍历结束，此时 nums2 中海油数据未拷贝完全，将其直接拷贝到 nums1 的前面，最后得到结果数组
     * 时间复杂度：O(m+n)
     * 代码
     *
     * 链接：https://leetcode-cn.com/problems/merge-sorted-array/solution/hua-jie-suan-fa-88-he-bing-liang-ge-you-xu-shu-zu-/
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int len1 = m - 1;
        int len2 = n - 1;
        int len = m + n - 1;
        while (len1 >= 0 && len2 >= 0) {
            // 注意--符号在后面，表示先进行计算再减1，这种缩写缩短了代码
            nums1[len--] = nums1[len1] > nums2[len2] ? nums1[len1--] : nums2[len2--];
        }
        // 表示将nums2数组从下标0位置开始，拷贝到nums1数组中，从下标0位置开始，长度为len2+1
        System.arraycopy(nums2, 0, nums1, 0, len2 + 1);
    }

    public void merge2(int[] nums1, int m, int[] nums2, int n) {
        int t1 = m - 1, t2 = n - 1, t = m + n - 1;
        while (t1 != -1 && t2 != -1) nums1[t--] = nums1[t1] >= nums2[t2] ? nums1[t1--] : nums2[t2--];
        while (t2 != -1) nums1[t--] = nums2[t2--];
    }

    /**
     * 【宫水三叶】一题多解：「双指针」&「先合再排」&「原地合并」解法 ...
     * https://leetcode-cn.com/problems/merge-sorted-array/solution/gong-shui-san-xie-yi-ti-san-jie-shuang-z-47gj/
     * *****************************************************************************************************************
     * 双指针（额外空间）
     * 一个简单的做法是，创建一个和 nums1 等长的数组 arr，使用双指针将 num1 和 nums2 的数据迁移到 arr。最后再将 arr 复制到 nums1 中。
     * 时间复杂度：O(m + n)
     * 空间复杂度：O(m + n)
     */
    public void merge3(int[] nums1, int m, int[] nums2, int n) {
        int total = m + n;
        int[] arr = new int[total];
        int idx = 0;
        for (int i = 0, j = 0; i < m || j < n; ) {
            if (i < m && j < n) {
                arr[idx++] = nums1[i] < nums2[j] ? nums1[i++] : nums2[j++];
            } else if (i < m) {
                arr[idx++] = nums1[i++];
            } else if (j < n) {
                arr[idx++] = nums2[j++];
            }
        }
        System.arraycopy(arr, 0, nums1, 0, total);
    }

    /**
     * 先合并再排序
     * 我们还可以将 nums2 的内容先迁移到 nums1 去，再对 nums1 进行排序。
     * 时间复杂度：O((m+n)log(m+n))
     * 空间复杂度：O(1)
     * PS. Java 中的 sort 排序是一个综合排序。包含插入/双轴快排/归并/timsort，这里假定 Arrays.sort 使用的是「双轴快排」，并忽略递归带来的空间开销。
     */
    public void merge4(int[] nums1, int m, int[] nums2, int n) {
        System.arraycopy(nums2, 0, nums1, m, n);
        Arrays.sort(nums1);
    }


    /**
     * 66. 加一（谷歌、字节跳动、Facebook 在半年内面试中考过）
     * 给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
     * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
     * 你可以假设除了整数 0 之外，这个整数不会以零开头。
     * 示例 1：
     *
     * 输入：digits = [1,2,3]
     * 输出：[1,2,4]
     * 解释：输入数组表示数字 123。
     * 示例 2：
     *
     * 输入：digits = [4,3,2,1]
     * 输出：[4,3,2,2]
     * 解释：输入数组表示数字 4321。
     * 示例 3：
     *
     * 输入：digits = [0]
     * 输出：[1]
     * *****************************************************************************************************************
     * 【数组】加一
     * 解题思路
     * 根据分析题意，我们总结一下可能会遇到的问题：
     *
     * 当前位是否需要进位？
     *
     * 36 + 1 = 37     // 无需进位
     * 19 + 1 = 20     // 存在进位
     * 数字长度是否会改变？
     *
     * 18 + 1 = 19     // 长度不变
     * 99 + 1 = 100    // 长度改变
     * 有了上述的分析，我们就好解决问题了，流程如下：
     *
     * 首先将数字转为数组形式，便于数据处理。
     * 数组末位数字加 11，然后开始倒序遍历数组。
     * 判断当前位数字是否需要进位，做出对应处理。
     * 遍历数组完毕，若还存在进位，则数组首位添加 11 即可。
     *
     * 作者：demigodliu
     * 链接：https://leetcode-cn.com/problems/plus-one/solution/shu-zu-jia-yi-by-demigodliu-m2c6/
     * *****************************************************************************************************************
     * 秒杀
     * 袁厨的算法小屋
     * 我们思考一下，加一的情况一共有几种情况呢？是不是有以下三种情况
     * 见链接
     * 则我们根据什么来判断属于第几种情况呢？
     * 我们可以根据当前位 余10来判断，这样我们就可以区分属于第几种情况了，大家直接看代码吧，很容易理解的。
     * https://leetcode-cn.com/problems/plus-one/solution/nu-peng-you-du-neng-kan-dong-de-ti-jie-b-p2zw/
     */
    public int[] plusOne(int[] digits) {
        //获取长度
        int len = digits.length;
        for (int i = len - 1; i >= 0; i--) {
            digits[i] = (digits[i] + 1) % 10;
            //第一种和第二种情况，如果此时某一位不为 0 ，则直接返回即可。
            if (digits[i] != 0) {
                return digits;
            }

        }
        //第三种情况，因为数组初始化每一位都为0，我们只需将首位设为1即可
        int[] arr = new int[len + 1];
        arr[0] = 1;
        return arr;
    }



    public static void main(String[] args) {
        int[] height = new int[]{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
        int[] height2 = new int[]{4, 2, 0, 3, 2, 5};
        Week1 week1 = new Week1();
        int trap11 = week1.trap(height);
        int trap12 = week1.trap(height2);
        int trap21 = week1.trap2(height);
        int trap22 = week1.trap2(height2);
        int trap31 = week1.trap3(height);
        int trap32 = week1.trap3(height2);
        int trap41 = week1.trap4(height);
        int trap42 = week1.trap4(height2);
        int trap51 = week1.trap5(height);
        int trap52 = week1.trap5(height2);
        int trap61 = week1.trap6(height);
        int trap62 = week1.trap6(height2);
        // 双指针
        int trap71 = week1.trap7(height);
        int trap72 = week1.trap7(height2);
        int trap81 = week1.trap8(height);
        int trap82 = week1.trap8(height2);
        System.out.println("trap11=" + trap11 + "   trap12=" + trap12 + "   trap21=" + trap21 + "   trap22=" + trap22 + "   trap31=" + trap31 + "   trap32=" + trap32 + "   trap41=" + trap41 + "   trap42=" + trap42 + "   trap51=" + trap51 + "   trap52=" + trap52 + "   trap61=" + trap61 + "   trap62=" + trap62 + "   trap71=" + trap71 + "   trap72=" + trap72 + "   trap81=" + trap81 + "   trap82=" + trap82);
        /**
         * @suan-tou-wang-ba 弱弱问一句，AC是什么意思
         * @代码搬运工 能够通过的代码
         * @代码搬运工 TLE 就是超时的代码
         * @代码搬运工 如果刷英文版的话会注意到代码通过显示的是Accepted。AC似乎就是这个词的缩写吧(前两个字母)。
         * @suan-tou-wang-ba 清晰易懂 感谢
         *
         * 双指针那块那个for循环给我看懵了，后来想了一下，把for循环改成while (left<=right)，然后就好懂了
         */

        // 盛最多水的容器
        int[] container = new int[]{60, 50, 80, 90, 70, 85, 65};
        System.out.println("maxArea = " + week1.maxArea(container) + "  maxArea2 = " + week1.maxArea2(container) + "    maxArea3 = " + week1.maxArea3(container));

        // 移动零
        int[] moveZeroArray = new int[]{0, 1, 0, 3, 12, 5, 0, 6};
        week1.moveZeroes(moveZeroArray);

        // climbStairs
        int steps = 5;
        int countWay = week1.climbStairs(steps);
        int countWay2 = week1.climbStairs2(steps);
        int countWay3 = week1.climbStairs3(steps);
        int countWay4 = week1.climbStairs4(steps);
        int countWay5 = week1.climbStairs5(steps);
        int countWay6 = week1.climbStairs6(steps);
        System.out.println("climbStairs = " + countWay + "  climbStairs2 = " + countWay2 + "  climbStairs3 = " + countWay3 + "  climbStairs4 = " + countWay4 + "  climbStairs5 = " + countWay5 + "  climbStairs6 = " + countWay6);

        // twoSum
        int[] twoSumArray = new int[]{2, 7, 11, 15};
        int target = 13;
        int[] indexArray = week1.twoSum(twoSumArray, target);
        for (int i = 0; i < indexArray.length; i++) {
            System.out.println("indexArray[" + i + "] = " + indexArray[i] + "  twoSumArray[" + indexArray[i] + "] = " + twoSumArray[indexArray[i]]);
        }
        int[] indexArray2 = week1.twoSum2(twoSumArray, target);
        for (int i = 0; i < indexArray2.length; i++) {
            System.out.println("indexArray2[" + i + "] = " + indexArray2[i] + "  twoSumArray[" + indexArray2[i] + "] = " + twoSumArray[indexArray2[i]]);
        }
        int[] indexArray3 = week1.twoSum3(twoSumArray, target);
        for (int i = 0; i < indexArray3.length; i++) {
            System.out.println("indexArray3[" + i + "] = " + indexArray3[i] + "  twoSumArray[" + indexArray3[i] + "] = " + twoSumArray[indexArray3[i]]);
        }

        // threeSum
        int[] threeSumArray = new int[]{-1, 0, 1, 2, -1, -4};
        List<List<Integer>> zeroSumList = week1.threeSum(threeSumArray);
        List<List<Integer>> zeroSumList2 = week1.threeSum2(threeSumArray);
        List<List<Integer>> zeroSumList3 = week1.threeSum3(threeSumArray);
        System.out.println("zeroSumList = " + zeroSumList + ";   zeroSumList2 = " + zeroSumList2 + ";    zeroSumList3 = " + zeroSumList3);

        // 删除排序数组中的重复项
        int[] duplicateArray = new int[]{0, 0, 1, 1, 1, 2, 2, 3, 3, 4};
        int length = week1.removeDuplicates(duplicateArray);
        System.out.println("removeDuplicatesLength = " + length);

        // 反转链表
    }
























}
