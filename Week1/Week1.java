package algorithm;

import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

/**
 * @ClassName Week1
 * @Description
 * @Author Administrator
 * @Date 2020/11/22  5:09
 * @Version 1.0
 **/
public class Week1 {

    // 1.用 add first 或 add last 这套新的 API 改写 Deque 的代码
    public  void overrideDeque() {
        Deque<String> deque = new LinkedList<String>();
        // push()
        deque.addFirst("a");
        deque.addFirst("b");
        deque.addFirst("c");
        deque.addLast("1");
        System.out.println(deque);

    }

    // 2.分析 Queue 和 Priority Queue 的源码

    // 3.删除排序数组中的重复项（Facebook、字节跳动、微软在半年内面试中考过）
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        int i = 0;
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] != nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }

    // 4.旋转数组（微软、亚马逊、PayPal 在半年内面试中考过）

    // 盛最多水的容器（腾讯、百度、字节跳动在近半年内面试常考）
    // 思路:双指针法左右俩端趋近
    // 给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点(i,ai) 。在坐标内画 n 条垂直线，垂直线 i的两个端点分别为(i,ai) 和 (i, 0) 。找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。
    // 说明：你不能倾斜容器。
    public int maxArea(int[] height) {
        int i = 0, j = height.length - 1, res = 0;
        while(i < j){
            // System.out.println("height[i]="+height[i]);
            // System.out.println("height[i++]="+height[i++]);
            // System.out.println("height[j]="+height[j]);
            // System.out.println("height[j--]="+height[j--]);
            res = height[i] < height[j] ?
                    Math.max(res, (j - i) * height[i++]):
                    Math.max(res, (j - i) * height[j--]);
        }
        return res;
    }
    // 8.移动零（Facebook、亚马逊、苹果在半年内面试中考过）
    // 思路:操作index;j记录到非0元素的位置在什么地方
    // 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
    public void moveZeroes(int[] nums) {
        int j =0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[j]=nums[i];
                if(i!=j){
                    nums[i]=0;
                }
                System.out.println("nums["+i+"]="+nums[i]+"  nums["+j+"]="+nums[j]);
                j++;
            }
        }
    }

    // 8.移动零(解法2:互换位置swap函数)
    // public void moveZeroes2(int[] nums) {
    //     for (int lastNonZeroFoundAt=0, cur = 0; cur < nums.length; cur++) {
    //         if (nums[cur] != 0) {
    //             swap(nums[lastNonZeroFoundAt++], nums[cur]);
    //         }
    //     }
    // }

    // 8.移动零(解法3: 遍历整个数组,遇0删除,在列表最后添加0)

    // --------------------------------------------------------------------------------
    // 用 add first 或 add last 这套新的 API 改写 Deque 的代码
    // 分析 Queue 和 Priority Queue 的源码
    // 删除排序数组中的重复项（Facebook、字节跳动、微软在半年内面试中考过）
    // 旋转数组（微软、亚马逊、PayPal 在半年内面试中考过）
    // 合并两个有序链表（亚马逊、字节跳动在半年内面试常考）
    // 合并两个有序数组（Facebook 在半年内面试常考）
    // 两数之和（亚马逊、字节跳动、谷歌、Facebook、苹果、微软在半年内面试中高频常考）
    // 移动零（Facebook、亚马逊、苹果在半年内面试中考过）
    // 加一（谷歌、字节跳动、Facebook 在半年内面试中考过）
    //
    // 设计循环双端队列（Facebook 在 1 年内面试中考过）
    // 接雨水（亚马逊、字节跳动、高盛集团、Facebook 在半年内面试常考）

}
