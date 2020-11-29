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

    // 盛最多水的容器（腾讯、百度、字节跳动在近半年内面试常考）[1,8,6,2,5,4,8,3,7]
    // 思路:双指针法左右俩端趋近即左右夹逼
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
    // 双指针法的第二钟解法
    public int maxArea2(int[] height) {
        int max =0 ;
        for (int i = 0 , j = height.length-1 ; i<j ;) {
            int minHeight = height[i] < height[j] ? height[i++] : height[j--];
            // int area = (j-i+1) * minHeight;
            // max = Math.max(max , area);
            max = Math.max(max , (j-i+1) * minHeight);
        }
        return max;
    }
    // 7.两数之和（亚马逊、字节跳动、谷歌、Facebook、苹果、微软在半年内面试中高频常考）nums = [2, 7, 11, 15], target = 9
    // 给定一个整数数组 nums和一个目标值 target，请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
    // 方法1:暴力解法  https://leetcode-cn.com/problems/two-sum/solution/liang-shu-zhi-he-by-leetcode-solution/
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
    // 方法2:
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
    // 方法3:超哥
    public int[] twoSum3(int[] nums, int target) {
        int[] a = new int[2];
        for (int i = 0; i < nums.length -1; i++) {
            for (int j = i+1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) {
                    a[0]=i;
                    a[1]=j;
                    return a;
                }
            }
        }
        return new int[0];
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
    public void moveZeroes2(int[] nums) {
        for (int lastNonZeroFoundAt=0, cur = 0; cur < nums.length; cur++) {
            if (nums[cur] != 0) {
                swap(nums[lastNonZeroFoundAt++], nums[cur]);
            }
        }
    }

    private void swap(int num, int num1) {

    }

    // 8.移动零(解法3: 遍历整个数组,遇0删除,在列表最后添加0)


    // 10.设计循环双端队列（Facebook 在 1 年内面试中考过）
    // 设计实现双端队列。
    // 你的实现需要支持以下操作：
    // MyCircularDeque(k)：构造函数,双端队列的大小为k。
    // insertFront()：将一个元素添加到双端队列头部。 如果操作成功返回 true。
    // insertLast()：将一个元素添加到双端队列尾部。如果操作成功返回 true。
    // deleteFront()：从双端队列头部删除一个元素。 如果操作成功返回 true。
    // deleteLast()：从双端队列尾部删除一个元素。如果操作成功返回 true。
    // getFront()：从双端队列头部获得一个元素。如果双端队列为空，返回 -1。
    // getRear()：获得双端队列的最后一个元素。如果双端队列为空，返回 -1。
    // isEmpty()：检查双端队列是否为空。
    // isFull()：检查双端队列是否满了。

    // 11.接雨水（亚马逊、字节跳动、高盛集团、Facebook 在半年内面试常考）
    // 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
    // 输入：height = [4,2,0,3,2,5]
    // 输出：9
    // 提示
    // n == height.length
    // 0 <= n <= 3 * 104
    // 0 <= height[i] <= 105
    public int trap(int[] height) {
        return 2;
    }
    // --------------------------------------------------------------------------------
    // 1.用 add first 或 add last 这套新的 API 改写 Deque 的代码
    // 2.分析 Queue 和 Priority Queue 的源码
    // 3.删除排序数组中的重复项（Facebook、字节跳动、微软在半年内面试中考过）
    // 4.旋转数组（微软、亚马逊、PayPal 在半年内面试中考过）
    // 5.合并两个有序链表（亚马逊、字节跳动在半年内面试常考）
    // 6.合并两个有序数组（Facebook 在半年内面试常考）
    // 7.两数之和（亚马逊、字节跳动、谷歌、Facebook、苹果、微软在半年内面试中高频常考）
    // 8.移动零（Facebook、亚马逊、苹果在半年内面试中考过）
    // 9.加一（谷歌、字节跳动、Facebook 在半年内面试中考过）
    // 10.设计循环双端队列（Facebook 在 1 年内面试中考过）
    // 11.接雨水（亚马逊、字节跳动、高盛集团、Facebook 在半年内面试常考）
    // --------------------------------------------------------------------------------
    // Array 实战题目
    // 1.两数之和（近半年内，字节跳动在面试中考查此题达到 152 次）
    // 2.盛最多水的容器（腾讯、百度、字节跳动在近半年内面试常考）
    // 3.移动零（华为、字节跳动在近半年内面试常考）
    // 4.爬楼梯（阿里巴巴、腾讯、字节跳动在半年内面试常考）
    // 三数之和（国内、国际大厂历年面试高频老题）
    // 1.Linked List 实战题目
    // 2.反转链表（字节跳动、亚马逊在半年内面试常考）
    // 3.两两交换链表中的节点（阿里巴巴、字节跳动在半年内面试常考）
    // 4.环形链表（阿里巴巴、字节跳动、腾讯在半年内面试常考）
    // 5.环形链表 II
    // 6.K 个一组翻转链表（字节跳动、猿辅导在半年内面试常考）
    // --------------------------------------------------------------------------------
    // 4.爬楼梯（阿里巴巴、腾讯、字节跳动在半年内面试常考）
    // 思路:找最近重复子问题(通项公式)(泛化)
    // f(n) = f(n-1) + f(n-2);Fibonacci数列
    public int climbStairs(int n) {
        if(n <= 2){return n;}
        int f1 = 1, f2 = 2, f3=3;
        for(int i = 3; i <= n; i++){
            f3 = f1 + f2;
            f1 = f2;
            f2 = f3;
        }
        return f3;
    }
    // 动态规划思路： 要考虑第爬到第n阶楼梯时候可能是一步，也可能是两步。
    // 1.计算爬上n-1阶楼梯的方法数量。因为再爬1阶就到第n阶
    // 2.计算爬上n-2阶楼梯体方法数量。因为再爬2阶就到第n阶 那么f(n)=f(n-1)+f(n-2);
    // 为什么不是f(n)=f(n-1)+1+f(n-2)+2呢，因为f(n)是爬楼梯方法数量，不是爬到n阶楼梯的步数
    public int climbStairs2(int n) {
        if(n==0||n==1)
            return n;
        int[] bp = new int[n];
        bp[0]=1;
        bp[1]=2;
        for(int i=2;i<n;i++)
        {
            bp[i]=bp[i-1]+bp[i-2];
        }
        return bp[n-1];
    }
    // f(n) = f(n-1) + f(n-2);Fibonacci数列
    public int climbStairs3(int n) {
        if (n <= 2) return n;
        return climbStairs(n - 1) + climbStairs(n - 2);
    }
    // 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
    // https://leetcode-cn.com/problems/climbing-stairs/comments/
    // 长度较短的有限集合的解，可直接返回值，自己学习算法最终的目的还是为了更好地解决问题。警醒自己不要沉迷于算法的精妙而忽视实际情况，上了很好的一课(switch case)
    // 第n个台阶只能从第n-1或者n-2个上来。到第n-1个台阶的走法 + 第n-2个台阶的走法 = 到第n个台阶的走法，已经知道了第1个和第2个台阶的走法，一路加上去。
    // https://leetcode-cn.com/problems/climbing-stairs/solution/pa-lou-ti-by-leetcode-solution/
    // 方法三：通项公式
    public int climbStairs4(int n) {
        double sqrt5 = Math.sqrt(5);
        double fibn = Math.pow((1 + sqrt5) / 2, n + 1) - Math.pow((1 - sqrt5) / 2, n + 1);
        return (int)(fibn / sqrt5);
    }

    // 三数之和（国内、国际大厂历年面试高频老题）
    // 给你一个包含 n 个整数的数组nums，判断nums中是否存在三个元素 a，b，c ，使得a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。注意：答案中不可以包含重复的三元组。nums = [-1, 0, 1, 2, -1, -4]
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> lists = new ArrayList<>();
        //排序
        Arrays.sort(nums);
        //双指针
        int len = nums.length;
        for(int i = 0;i < len;++i) {
            if(nums[i] > 0) return lists;
            if(i > 0 && nums[i] == nums[i-1]) continue;
            int curr = nums[i];
            int L = i+1, R = len-1;
            while (L < R) {
                int tmp = curr + nums[L] + nums[R];
                if(tmp == 0) {
                    List<Integer> list = new ArrayList<>();
                    list.add(curr);
                    list.add(nums[L]);
                    list.add(nums[R]);
                    lists.add(list);
                    while(L < R && nums[L+1] == nums[L]) ++L;
                    while (L < R && nums[R-1] == nums[R]) --R;
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





















}
