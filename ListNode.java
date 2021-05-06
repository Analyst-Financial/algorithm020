package algorithm;

/**
 * @ClassName ListNode
 * @Description TODO
 * @Author Administrator
 * @Date 2021/3/5  16:42
 * @Version 1.0
 **/
public class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}
