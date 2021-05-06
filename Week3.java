package algorithm;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @ClassName Week3
 * @Description
 * @Author Administrator
 * @Date 2020/11/30  22:06
 * @Version 1.0
 **/
public class Week3 {


    /**
     *  第3周 第7课 | 泛型递归、树的递归
     *  1. 递归的实现、特性以及思维要点
     *  2. 实战题目解析：爬楼梯、括号生成等问题
     *  第3周 第8课 | 分治、回溯
     *  1. 分治、回溯的实现和特性
     *  2. 实战题目解析：Pow(x,n)、子集
     *  3. 实战题目解析：电话号码的字母组合、N皇后
     */

    /**
     * 前言
     *
     * 二叉树前序遍历的顺序为：
     *
     * 先遍历根节点；
     * 随后递归地遍历左子树；
     * 最后递归地遍历右子树。
     *
     * 二叉树中序遍历的顺序为：
     *
     * 先递归地遍历左子树；
     * 随后遍历根节点；
     * 最后递归地遍历右子树。
     * 在「递归」地遍历某个子树的过程中，我们也是将这颗子树看成一颗全新的树，按照上述的顺序进行遍历。挖掘「前序遍历」和「中序遍历」的性质，我们就可以得出本题的做法。
     */

    /**
     * 4种解法秒杀TopK（快排/堆/二叉搜索树/计数排序）❤️
     * 作者：sweetiee
     * 链接：https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/solution/3chong-jie-fa-miao-sha-topkkuai-pai-dui-er-cha-sou/
     * *****************************************************************************************************************
     * 解题思路：
     * 对于经典TopK问题，本文给出 4 种通用解决方案。
     * 解题思路：
     * 一、用快排最最最高效解决 TopK 问题：
     * 二、大根堆(前 K 小) / 小根堆（前 K 大),Java中有现成的 PriorityQueue，实现起来最简单：
     * 三、二叉搜索树也可以 解决 TopK 问题哦
     * 四、数据范围有限时直接计数排序就行了：
     * *****************************************************************************************************************
     * 一、用快排最最最高效解决 TopK 问题：O(N)
     * 注意找前 K 大/前 K 小问题不需要对整个数组进行 O(NlogN) 的排序！
     * 例如本题，直接通过快排切分排好第 K 小的数（下标为 K-1），那么它左边的数就是比它小的另外 K-1 个数啦～
     * 下面代码给出了详细的注释，没啥好啰嗦的，就是快排模版要记牢哈～
     *
     *
     * 作者：sweetiee
     * 链接：https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/solution/3chong-jie-fa-miao-sha-topkkuai-pai-dui-er-cha-sou/
     */
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        // 最后一个参数表示我们要找的是下标为k-1的数
        return quickSearch(arr, 0, arr.length - 1, k - 1);
    }

    private int[] quickSearch(int[] nums, int lo, int hi, int k) {
        // 每快排切分1次，找到排序后下标为j的元素，如果j恰好等于k就返回j以及j左边所有的数；
        int j = partition(nums, lo, hi);
        if (j == k) {
            return Arrays.copyOf(nums, j + 1);
        }
        // 否则根据下标j与k的大小关系来决定继续切分左段还是右段。
        return j > k ? quickSearch(nums, lo, j - 1, k) : quickSearch(nums, j + 1, hi, k);
    }

    // 快排切分，返回下标j，使得比nums[j]小的数都在j的左边，比nums[j]大的数都在j的右边。
    private int partition(int[] nums, int lo, int hi) {
        int v = nums[lo];
        int i = lo, j = hi + 1;
        while (true) {
            while (++i <= hi && nums[i] < v) ;
            while (--j >= lo && nums[j] > v) ;
            if (i >= j) {
                break;
            }
            int t = nums[j];
            nums[j] = nums[i];
            nums[i] = t;
        }
        nums[lo] = nums[j];
        nums[j] = v;
        return j;
    }

    /**
     * 快排切分时间复杂度分析：
     * 因为我们是要找下标为k的元素，第一次切分的时候需要遍历整个数组 (0 ~ n) 找到了下标是 j 的元素，假如 k 比 j 小的话，那么我们下次切分只要遍历数组 (0~k-1)的元素就行啦，
     * 反之如果 k 比 j 大的话，那下次切分只要遍历数组 (k+1～n) 的元素就行啦，总之可以看作每次调用 partition 遍历的元素数目都是上一次遍历的 1/2，因此时间复杂度是 N + N/2 + N/4 + ... + N/N = 2N,
     * 因此时间复杂度是 O(N)。
     *
     * 二、大根堆(前 K 小) / 小根堆（前 K 大),Java中有现成的 PriorityQueue，实现起来最简单：O(NlogK)
     * 本题是求前 K 小，因此用一个容量为 K 的大根堆，每次 poll 出最大的数，那堆中保留的就是前 K 小啦（注意不是小根堆！小根堆的话需要把全部的元素都入堆，那是 O(NlogN)，就不是 O(NlogK)啦～～）
     * 这个方法比快排慢，但是因为 Java 中提供了现成的 PriorityQueue（默认小根堆），所以实现起来最简单，没几行代码～
     *
     * 保持堆的大小为K，然后遍历数组中的数字，遍历的时候做如下判断：
     * 1. 若目前堆的大小小于K，将当前数字放入堆中。
     * 2. 否则判断当前数字与大根堆堆顶元素的大小关系，如果当前数字比大根堆堆顶还大，这个数就直接跳过；反之如果当前数字比大根堆堆顶小，先poll掉堆顶，再将该数字放入堆中。
     *
     * 作者：sweetiee
     * 链接：https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/solution/3chong-jie-fa-miao-sha-topkkuai-pai-dui-er-cha-sou/
     */
    public int[] getLeastNumbers2(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        // 默认是小根堆，实现大根堆需要重写一下比较器。
        Queue<Integer> pq = new PriorityQueue<>((v1, v2) -> v2 - v1);
        for (int num : arr) {
            if (pq.size() < k) {
                pq.offer(num);
            } else if (num < pq.peek()) {
                pq.poll();
                pq.offer(num);
            }
        }

        // 返回堆中的元素
        int[] res = new int[pq.size()];
        int idx = 0;
        for (int num : pq) {
            res[idx++] = num;
        }
        return res;
    }

    /**
     * 三、二叉搜索树也可以 O(NlogK)解决 TopK 问题哦
     * BST 相对于前两种方法没那么常见，但是也很简单，和大根堆的思路差不多～
     * 要提的是，与前两种方法相比，BST 有一个好处是求得的前K大的数字是有序的。
     *
     * 因为有重复的数字，所以用的是 TreeMap 而不是 TreeSet（有的语言的标准库自带 TreeMultiset，也是可以的）。
     *
     * TreeMap的key 是数字，value 是该数字的个数。
     * 我们遍历数组中的数字，维护一个数字总个数为 K 的 TreeMap：
     * 1.若目前 map 中数字个数小于 K，则将 map 中当前数字对应的个数 +1；
     * 2.否则，判断当前数字与 map 中最大的数字的大小关系：
     * 若当前数字大于等于 map 中的最大数字，就直接跳过该数字；
     * 若当前数字小于 map 中的最大数字，则将 map 中当前数字对应的个数 +1，并将 map 中最大数字对应的个数减 1。
     *
     * 作者：sweetiee
     * 链接：https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/solution/3chong-jie-fa-miao-sha-topkkuai-pai-dui-er-cha-sou/
     */
    public int[] getLeastNumbers3(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        // TreeMap的key是数字, value是该数字的个数。
        // cnt表示当前map总共存了多少个数字。
        TreeMap<Integer, Integer> map = new TreeMap<>();
        int cnt = 0;
        for (int num : arr) {
            // 1. 遍历数组，若当前map中的数字个数小于k，则map中当前数字对应个数+1
            if (cnt < k) {
                map.put(num, map.getOrDefault(num, 0) + 1);
                cnt++;
                continue;
            }
            // 2. 否则，取出map中最大的Key（即最大的数字), 判断当前数字与map中最大数字的大小关系：
            //    若当前数字比map中最大的数字还大，就直接忽略；
            //    若当前数字比map中最大的数字小，则将当前数字加入map中，并将map中的最大数字的个数-1。
            Map.Entry<Integer, Integer> entry = map.lastEntry();
            if (entry.getKey() > num) {
                map.put(num, map.getOrDefault(num, 0) + 1);
                if (entry.getValue() == 1) {
                    map.pollLastEntry();
                } else {
                    map.put(entry.getKey(), entry.getValue() - 1);
                }
            }

        }

        // 最后返回map中的元素
        int[] res = new int[k];
        int idx = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int freq = entry.getValue();
            while (freq-- > 0) {
                res[idx++] = entry.getKey();
            }
        }
        return res;
    }

    /**
     * 四、数据范围有限时直接计数排序就行了：O(N)
     * *****************************************************************************************************************
     * 最后
     * 雷同题目 215. 数组中的第K个最大元素 常考哦～
     */
    public int[] getLeastNumbers4(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        // 统计每个数字出现的次数
        int[] counter = new int[10001];
        for (int num : arr) {
            counter[num]++;
        }
        // 根据counter数组从头找出k个数作为返回结果
        int[] res = new int[k];
        int idx = 0;
        for (int num = 0; num < counter.length; num++) {
            while (counter[num]-- > 0 && idx < k) {
                res[idx++] = num;
            }
            if (idx == k) {
                break;
            }
        }
        return res;
    }
    // ------------------------------------------------ 这 是 一 条 分 割 线 ----------------------------------------------
    // ------------------------------------------------ 这 是 一 条 分 割 线 ----------------------------------------------

    /**
     *  第3周 第8课 | 分治、回溯
     *  1. 分治、回溯的实现和特性
     *  2. 实战题目解析：Pow(x,n)、子集
     *  3. 实战题目解析：电话号码的字母组合、N皇后
     *  ****************************************************************************************************************
     *  分治代码模板
     *  括号生成问题
     *  Pow(x, n) （Facebook 在半年内面试常考）
     * 子集（Facebook、字节跳动、亚马逊在半年内面试中考过）
     * 多数元素 （亚马逊、字节跳动、Facebook 在半年内面试中考过）
     * 电话号码的字母组合（亚马逊在半年内面试常考）
     *  N 皇后（字节跳动、苹果、谷歌在半年内面试中考过）
     * *****************************************************************************************************************
     * 本周作业
     * 中等：
     * 二叉树的最近公共祖先（Facebook 在半年内面试常考）
     * 从前序与中序遍历序列构造二叉树（字节跳动、亚马逊、微软在半年内面试中考过）
     * 组合（微软、亚马逊、谷歌在半年内面试中考过）
     * 全排列（字节跳动在半年内面试常考）
     * 全排列 II （亚马逊、字节跳动、Facebook 在半年内面试中考过）
     *
     * 递归相关：
     * 指令计算器设计
     * 赛程表问题
     *
     * 回溯相关：
     * 单词转换
     */

    /**
     * 分治--回溯
     * 分治:
     * template:
     * 1.terminator    2.process(split your big problem)   3.drill down(subproblems) , merge(subresult)   4.reverse states
     *
     * x^n --> 2^10 : 2^5 --> (2^2)*2
     * pow(x,n)
     *  subproblem: subresult = pow(x,n/2)
     *  merge:
     *      if n % 2 == 1 {
     *          // odd
     *          result = subresult * subresult * x ;
     *      } else {
     *          // even
     *          result = subresult * subresult ;
     *      }
     *
     */

    /**
     * 22. 括号生成 (字节跳动在半年内面试中考过)
     * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
     * *****************************************************************************************************************
     * 思路:有规律啊，剩余左括号总数要小于等于右括号。 递归把所有符合要求的加上去就行了
     */
    List<String> res = new ArrayList<>();

    public List<String> generateParenthesis(int n) {
        if (n <= 0) {
            return res;
        }
        getParenthesis("", n, n);
        return res;
    }

    private void getParenthesis(String str, int left, int right) {
        if (left == 0 && right == 0) {
            res.add(str);
            return;
        }
        if (left == right) {
            // 剩余左右括号数相等，下一个只能用左括号
            getParenthesis(str + "(", left - 1, right);
        } else if (left < right) {
            // 剩余左括号小于右括号，下一个可以用左括号也可以用右括号
            if (left > 0) {
                getParenthesis(str + "(", left - 1, right);
            }
            getParenthesis(str + ")", left, right - 1);
        }
    }

    /**
     * 超哥
     * left 随时可以加,只要别超标
     * right 左括号个数>右括号个数
     */
    private ArrayList<String> result;

    public List<String> generateParenthesis2(int n) {
        result = new ArrayList<>();
        _generate(0, 0, n, "");
        return result;
    }

    private void _generate(int left, int right, int n, String s) {
        // terminator
        if (left == n && right == n) {
            // System.out.println(s);
            result.add(s);
            return;
        }
        // process current logic

        // drill down
        String s1 = s + "(";
        String s2 = s + ")";
        if (left < n) {
            _generate(left + 1, right, n, s1);
        }
        if (left > right) {
            _generate(left, right + 1, n, s2);
        }
        // reverse states
    }

    /**
     * 50. Pow(x, n)（Facebook 在半年内面试常考）
     * 实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，x^n）。
     *
     * 示例 1：
     * 输入：x = 2.00000, n = 10
     * 输出：1024.00000
     * *****************************************************************************************************************
     *
     */
    public double myPow(double x, int n) {
        if (n < 0) return 1.0 / myPow(x, -n);
        double res = 1.0;
        for (int i = n; i != 0; i /= 2) {
            if (i % 2 != 0) {
                res *= x;
            }
            x *= x;
        }
        return res;
    }

    /**
     * 官方解法
     * https://leetcode-cn.com/problems/powx-n/solution/powx-n-by-leetcode-solution/
     *
     * 方法一：快速幂 + 递归
     */
    public double quickMul(double x, long N) {
        if (N == 0) {
            return 1.0;
        }
        double y = quickMul(x, N / 2);
        return N % 2 == 0 ? y * y : y * y * x;
    }

    public double myPow2(double x, int n) {
        long N = n;
        return N >= 0 ? quickMul(x, N) : 1.0 / quickMul(x, -N);
    }

    /**
     * 方法二：快速幂 + 迭代
     */
    double quickMul2(double x, long N) {
        double ans = 1.0;
        // 贡献的初始值为 x
        double x_contribute = x;
        // 在对 N 进行二进制拆分的同时计算答案
        while (N > 0) {
            if (N % 2 == 1) {
                // 如果 N 二进制表示的最低位为 1，那么需要计入贡献
                ans *= x_contribute;
            }
            // 将贡献不断地平方
            x_contribute *= x_contribute;
            // 舍弃 N 二进制表示的最低位，这样我们每次只要判断最低位即可
            N /= 2;
        }
        return ans;
    }

    public double myPow3(double x, int n) {
        long N = n;
        return N >= 0 ? quickMul2(x, N) : 1.0 / quickMul2(x, -N);
    }

    /**
     * 78. 子集（Facebook、字节跳动、亚马逊在半年内面试中考过）
     * 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
     * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
     * *****************************************************************************************************************
     * 方法一：递归法实现子集枚举
     * [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3], []]
     *
     *
     * 超哥的方法
     * def subsets(self, nums):
     *      result = [[]]
     *      for num in nums:
     *          newsets = []
     *          for subset in result:
     *              new_subset = subset + [num]
     *              newsets.append(new_subset)
     *      result.extend(newsets)
     * return result
     */
    public List<List<Integer>> subsets2(int[] nums) {
        List<List<Integer>> ans2 = new ArrayList<List<Integer>>();
        if (nums == null) {
            return ans2;
        }
        dfs(ans2, nums, new ArrayList<Integer>(), 0);
        return ans2;
    }

    public void dfs(List<List<Integer>> ans2, int[] nums, List<Integer> list, int index) {
        // terminator
        if (index == nums.length) {
            ans2.add(new ArrayList<Integer>(list));
            return;
        }
        // process
        dfs(ans2, nums, list, index + 1); // not pick the number at this index
        list.add(nums[index]);
        dfs(ans2, nums, list, index + 1); // pick the number at this index
        // reverse the current state
        list.remove(list.size() - 1);
    }

    /**
     * 思路二：
     * 逐个枚举，空集的幂集只有空集，每增加一个元素，让之前幂集中的每个集合，追加这个元素，就是新增的子集。
     * *****************************************************************************************************************
     * 循环枚举
     * [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
     */
    public static List<List<Integer>> enumerate(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        res.add(new ArrayList<Integer>());
        for (Integer n : nums) {
            int size = res.size();
            for (int i = 0; i < size; i++) {
                List<Integer> newSub = new ArrayList<Integer>(res.get(i));
                newSub.add(n);
                res.add(newSub);
            }
        }
        return res;
    }

    /**
     * 递归枚举
     */
    public static void recursion(int[] nums, int i, List<List<Integer>> res) {
        if (i >= nums.length) return;
        int size = res.size();
        for (int j = 0; j < size; j++) {
            List<Integer> newSub = new ArrayList<Integer>(res.get(j));
            newSub.add(nums[i]);
            res.add(newSub);
        }
        recursion(nums, i + 1, res);
    }

    /**
     * 思路三：
     * 集合中每个元素的选和不选，构成了一个满二叉状态树，比如，左子树是不选，右子树是选，从根节点、到叶子节点的所有路径，构成了所有子集。
     * 可以有前序、中序、后序的不同写法，结果的顺序不一样。本质上，其实是比较完整的中序遍历。
     * 链接：https://leetcode-cn.com/problems/subsets/solution/er-jin-zhi-wei-zhu-ge-mei-ju-dfssan-chong-si-lu-9c/
     * DFS，前序遍历
     */
    public static void preOrder(int[] nums, int i, ArrayList<Integer> subset, List<List<Integer>> res) {
        if (i >= nums.length) return;
        // 到了新的状态，记录新的路径，要重新拷贝一份
        subset = new ArrayList<Integer>(subset);

        // 这里
        res.add(subset);
        preOrder(nums, i + 1, subset, res);
        subset.add(nums[i]);
        preOrder(nums, i + 1, subset, res);
    }

    /**
     * DFS，中序遍历
     */
    public static void inOrder(int[] nums, int i, ArrayList<Integer> subset, List<List<Integer>> res) {
        if (i >= nums.length) return;
        subset = new ArrayList<Integer>(subset);

        inOrder(nums, i + 1, subset, res);
        subset.add(nums[i]);
        // 这里
        res.add(subset);
        inOrder(nums, i + 1, subset, res);
    }

    /**
     * DFS，后序遍历
     */
    public static void postOrder(int[] nums, int i, ArrayList<Integer> subset, List<List<Integer>> res) {
        if (i >= nums.length) return;
        subset = new ArrayList<Integer>(subset);

        postOrder(nums, i + 1, subset, res);
        subset.add(nums[i]);
        postOrder(nums, i + 1, subset, res);
        // 这里
        res.add(subset);
    }

    /**
     * 也可以左子树是选，右子树是不选，相当于，也可以前序、中序、后序的不同写法。程序实现上，需要用一个栈，元素入栈遍历左子树，（最近入栈的那个）元素出栈，遍历右子树。
     */
    public static void newPreOrder(int[] nums, int i, LinkedList<Integer> stack, List<List<Integer>> res) {
        if (i >= nums.length) return;
        stack.push(nums[i]);
        // 这里
        res.add(new ArrayList<Integer>(stack));
        newPreOrder(nums, i + 1, stack, res);
        stack.pop();
        newPreOrder(nums, i + 1, stack, res);
    }

    /**
     * A general approach to backtracking questions in Java (Subsets, Permutations, Combination Sum, Palindrome Partitioning)
     * This structure might apply to many other backtracking questions, but here I am just going to demonstrate Subsets, Permutations, and Combination Sum.
     *
     * Subsets : https://leetcode.com/problems/subsets/
     * *****************************************************************************************************************
     * https://leetcode.com/problems/subsets/discuss/27281/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partitioning)
     * 省略了很多方法，自己点开链接看
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        Arrays.sort(nums);
        backtrack(list, new ArrayList<>(), nums, 0);
        return list;
    }

    private void backtrack(List<List<Integer>> list, List<Integer> tempList, int[] nums, int start) {
        list.add(new ArrayList<>(tempList));
        for (int i = start; i < nums.length; i++) {
            tempList.add(nums[i]);
            backtrack(list, tempList, nums, i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }

    /**
     * 牛顿迭代法原理：
     * http://www.matrix67.com/blog/archives/361
     *
     * 牛顿迭代法代码：
     * Leetcode Sqrt(x):牛顿迭代法和Quake-III中的神奇方法
     * http://www.voidcn.com/article/p-eudisdmk-zm.html
     *
     * 自己看
     */


    // ------------------------------------------------ 这 是 一 条 分 割 线 ----------------------------------------------
    // ------------------------------------------------ 这 是 一 条 分 割 线 ----------------------------------------------

    /**
     * 169. 多数元素 （亚马逊、字节跳动、Facebook 在半年内面试中考过）
     * 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
     * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
     * *****************************************************************************************************************
     * Java-3种方法(计数法/排序法/摩尔投票法)
     * 最原始的思路
     * 遍历整个数组，对记录每个数值出现的次数(利用HashMap，其中key为数值，value为出现次数)；
     * 接着遍历HashMap中的每个Entry，寻找value值> nums.length / 2的key即可。
     *
     * 链接：https://leetcode-cn.com/problems/majority-element/solution/3chong-fang-fa-by-gfu-2/
     *
     * stream|merge
     */
    public int majorityElement(int[] nums) {
        Map<Integer, Long> map = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        long limit = nums.length >> 1;
        for (Map.Entry<Integer, Long> entry : map.entrySet())
            if (entry.getValue() > limit)
                return entry.getKey();
        return -1;
    }

    public int majorityElement2(int[] nums) {
        int limit = nums.length >> 1;
        HashMap<Integer, Integer> map = new HashMap<>(limit);
        for (int num : nums)
            map.merge(num, 1, (o_val, n_val) -> o_val + n_val);
        for (Map.Entry<Integer, Integer> entry : map.entrySet())
            if (entry.getValue() > limit)
                return entry.getKey();
        return -1;
    }



    /**
     * 排序思路
     * 既然数组中有出现次数> ⌊ n/2 ⌋的元素，那排好序之后的数组中，相同元素总是相邻的。
     * 即存在长度> ⌊ n/2 ⌋的一长串 由相同元素构成的连续子数组。
     * 举个例子：
     * 无论是1 1 1 2 3，0 1 1 1 2还是-1 0 1 1 1，数组中间的元素总是“多数元素”，毕竟它长度> ⌊ n/2 ⌋。
     *
     * 链接：https://leetcode-cn.com/problems/majority-element/solution/3chong-fang-fa-by-gfu-2/
     *
     *  Arrays.sort()|topK
     */
    public int majorityElement3(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length >> 1];
    }

    public int majorityElement4(int[] nums) {
        int len = (nums.length + 1) >> 1;
        PriorityQueue<Integer> pQueue = new PriorityQueue<>(len, Comparator.comparingInt(item -> -item));
        for (int num : nums) {
            pQueue.offer(num);
            if (pQueue.size() > len)
                pQueue.poll();
        }
        return pQueue.poll();
    }

    /**
     * 摩尔投票法思路
     * 候选人(cand_num)初始化为nums[0]，票数count初始化为1。
     * 当遇到与cand_num相同的数，则票数count = count + 1，否则票数count = count - 1。
     * 当票数count为0时，更换候选人，并将票数count重置为1。
     * 遍历完数组后，cand_num即为最终答案。
     *
     * 为何这行得通呢？
     * 投票法是遇到相同的则票数 + 1，遇到不同的则票数 - 1。
     * 且“多数元素”的个数> ⌊ n/2 ⌋，其余元素的个数总和<= ⌊ n/2 ⌋。
     * 因此“多数元素”的个数 - 其余元素的个数总和 的结果 肯定 >= 1。
     * 这就相当于每个“多数元素”和其他元素 两两相互抵消，抵消到最后肯定还剩余至少1个“多数元素”。
     *
     * 无论数组是1 2 1 2 1，亦或是1 2 2 1 1，总能得到正确的候选人。
     *
     * 链接：https://leetcode-cn.com/problems/majority-element/solution/3chong-fang-fa-by-gfu-2/
     * 理解了摩尔投票法后，良心推荐一道进阶的题目：求众数Ⅱ https://leetcode-cn.com/problems/majority-element-ii/
     */
    public int majorityElement5(int[] nums) {
        int cand_num = nums[0], count = 1;
        for (int i = 1; i < nums.length; ++i) {
            if (cand_num == nums[i])
                ++count;
            else if (--count == 0) {
                cand_num = nums[i];
                count = 1;
            }
        }
        return cand_num;
    }

    /**
     *  51. N 皇后（字节跳动、苹果、谷歌在半年内面试中考过）
     *  见 week 8
     */

    /**
     * 实战题目
     * 爬楼梯（阿里巴巴、腾讯、字节跳动在半年内面试常考）
     * 括号生成 (字节跳动在半年内面试中考过)
     * 翻转二叉树 (谷歌、字节跳动、Facebook 在半年内面试中考过)
     * 验证二叉搜索树（亚马逊、微软、Facebook 在半年内面试中考过）
     * 二叉树的最大深度（亚马逊、微软、字节跳动在半年内面试中考过）
     * 二叉树的最小深度（Facebook、字节跳动、谷歌在半年内面试中考过）
     * 二叉树的序列化与反序列化（Facebook、亚马逊在半年内面试常考）
     * *****************************************************************************************************************
     * 课后作业
     * 二叉树的最近公共祖先（Facebook 在半年内面试常考）
     * 从前序与中序遍历序列构造二叉树（字节跳动、亚马逊、微软在半年内面试中考过）
     * 组合（微软、亚马逊、谷歌在半年内面试中考过）
     * 全排列（字节跳动在半年内面试常考）
     * 全排列 II （亚马逊、字节跳动、Facebook 在半年内面试中考过）
     */


    /**
     * 236. 二叉树的最近公共祖先（Facebook 在半年内面试常考）
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     * *****************************************************************************************************************
     * 二叉树的最近公共祖先
     * 方法一：递归
     * 关键字:二叉树
     * 模式识别:树的问题通常可以用递归解决
     * 定义子问题：左右子树分别包括p,q
     * (lson && rson) || ((root.val == p.val || root.val == q.val) && (lson || rson))
     * 其中 lson 和 rson 分别代表 x 节点的左孩子和右孩子
     * (lson && rson)
     * 说明左子树和右子树均包含 p 节点或 q 节点，
     * 如果左子树包含的是 p 节点，那么右子树只能包含 q 节点，
     * 反之亦然，因为 p 节点和 q 节点都是不同且唯一的节点，
     * 因此如果满足这个判断条件即可说明 x 就是我们要找的最近公共祖先。
     * (x = p ∣∣ x = q) && (lson || rson)
     * 再来看第二条判断条件，这个判断条件即是考虑了 x 恰好是 p 节点或 q 节点且它的左子树或右子树有一个包含了另一个节点的情况，
     * 因此如果满足这个判断条件亦可说明 x 就是我们要找的最近公共祖先。
     *
     * https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/solution/er-cha-shu-de-zui-jin-gong-gong-zu-xian-by-leetc-2/
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        this.dfs(root, p, q);
        return this.treeNode;
    }

    private TreeNode treeNode;

    private boolean dfs(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return false;
        boolean lson = dfs(root.left, p, q);
        boolean rson = dfs(root.right, p, q);
        if ((lson && rson) || ((root.val == p.val || root.val == q.val) && (lson || rson))) {
            treeNode = root;
        }
        return lson || rson || (root.val == p.val || root.val == q.val);
    }

    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;//base case
        TreeNode left = lowestCommonAncestor2(root.left, p, q);//result
        TreeNode right = lowestCommonAncestor2(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        return root;//both left and right are not null, we found our result
    }



    /**
     * By Stefan Pochmann
     * https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/discuss/65225/4-lines-C%2B%2BJavaPythonRuby
     * So clever, the last line can be explained by
     * if(left == null && right == null) return null;
     * if(left != null && right != null) return root;
     * return left == null ? right : left;
     * <p>
     * if (left == null) return right;
     * else if (right == null) return left;
     * else return root;
     */
    public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor3(root.left, p, q);
        TreeNode right = lowestCommonAncestor3(root.right, p, q);
        return left == null ? right : right == null ? left : root;
    }



    /**
     * 105. 从前序与中序遍历序列构造二叉树（字节跳动、亚马逊、微软在半年内面试中考过）
     * 根据一棵树的前序遍历与中序遍历构造二叉树。
     * 注意:
     * 你可以假设树中没有重复的元素。
     * *****************************************************************************************************************
     * 详细通俗的思路分析，多解法
     * 解法一、递归
     * 先序遍历的顺序是根节点，左子树，右子树。中序遍历的顺序是左子树，根节点，右子树。
     * 所以我们只需要根据先序遍历得到根节点，然后在中序遍历中找到根节点的位置，它的左边就是左子树的节点，右边就是右子树的节点。
     * 生成左子树和右子树就可以递归的进行了。
     * 比如上图的例子，我们来分析一下。
     *
     * preorder = [3,9,20,15,7]
     * inorder = [9,3,15,20,7]
     * 首先根据 preorder 找到根节点是 3
     *
     * 然后根据根节点将 inorder 分成左子树和右子树
     * 左子树
     * inorder [9]
     *
     * 右子树
     * inorder [15,20,7]
     *
     * 把相应的前序遍历的数组也加进来
     * 左子树
     * preorder[9]
     * inorder [9]
     *
     * 右子树
     * preorder[20 15 7]
     * inorder [15,20,7]
     *
     * 现在我们只需要构造左子树和右子树即可，成功把大问题化成了小问题
     * 然后重复上边的步骤继续划分，直到 preorder 和 inorder 都为空，返回 null 即可
     *
     * 事实上，我们不需要真的把 preorder 和 inorder 切分了，只需要用分别用两个指针指向开头和结束位置即可。注意下边的两个指针指向的数组范围是包括左边界，不包括右边界。
     * 对于下边的树的合成。
     *
     * 链接：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by--22/
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTreeHelper(preorder, 0, preorder.length, inorder, 0, inorder.length);
    }

    private TreeNode buildTreeHelper(int[] preorder, int p_start, int p_end, int[] inorder, int i_start, int i_end) {
        // preorder 为空，直接返回 null
        if (p_start == p_end) {
            return null;
        }
        int root_val = preorder[p_start];
        TreeNode root = new TreeNode(root_val);
        //在中序遍历中找到根节点的位置
        int i_root_index = 0;
        for (int i = i_start; i < i_end; i++) {
            if (root_val == inorder[i]) {
                i_root_index = i;
                break;
            }
        }
        int leftNum = i_root_index - i_start;
        //递归的构造左子树
        root.left = buildTreeHelper(preorder, p_start + 1, p_start + leftNum + 1, inorder, i_start, i_root_index);
        //递归的构造右子树
        root.right = buildTreeHelper(preorder, p_start + leftNum + 1, p_end, inorder, i_root_index + 1, i_end);
        return root;
    }
    /**
     * 上边的代码很好理解，但存在一个问题，在中序遍历中找到根节点的位置每次都得遍历中序遍历的数组去寻找，参考这里 ，我们可以用一个HashMap把中序遍历数组的每个元素的值和下标存起来，这样寻找根节点的位置就可以直接得到了。
     * 链接：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by--22/
     */
    public TreeNode buildTree2(int[] preorder, int[] inorder) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        return buildTreeHelper(preorder, 0, preorder.length, inorder, 0, inorder.length, map);
    }

    private TreeNode buildTreeHelper(int[] preorder, int p_start, int p_end, int[] inorder, int i_start, int i_end, HashMap<Integer, Integer> map) {
        if (p_start == p_end) {
            return null;
        }
        int root_val = preorder[p_start];
        TreeNode root = new TreeNode(root_val);
        int i_root_index = map.get(root_val);
        int leftNum = i_root_index - i_start;
        root.left = buildTreeHelper(preorder, p_start + 1, p_start + leftNum + 1, inorder, i_start, i_root_index, map);
        root.right = buildTreeHelper(preorder, p_start + leftNum + 1, p_end, inorder, i_root_index + 1, i_end, map);
        return root;
    }

    /**
     * 本以为已经完美了，在 这里 又看到了令人眼前一亮的思路，就是 StefanPochmann 大神，经常逛 Discuss 一定会注意到他，拥有 3 万多的赞。
     * 他也发现了每次都得遍历一次去找中序遍历数组中的根节点的麻烦，但他没有用 HashMap就解决了这个问题，下边来说一下。
     * 用pre变量保存当前要构造的树的根节点，从根节点开始递归的构造左子树和右子树，in变量指向当前根节点可用数字的开头，然后对于当前pre有一个停止点stop，从in到stop表示要构造的树当前的数字范围。
     *
     * 代码很简洁，但如果细想起来真的很难理解了。
     * 把他的原话也贴过来吧。
     * Consider the example again. Instead of finding the 1 in inorder, splitting the arrays into parts and recursing on them, just recurse on the full remaining arrays and stop when you come across the 1 in inorder. That's what my above solution does. Each recursive call gets told where to stop, and it tells its subcalls where to stop. It gives its own root value as stopper to its left subcall and its parent`s stopper as stopper to its right subcall.
     * 本来很想讲清楚这个算法，但是各种画图，还是太难说清楚了。这里就画几个过程中的图，大家也只能按照上边的代码走一遍，理解一下了。
     *
     * 总之他的思想就是，不再从中序遍历中寻找根节点的位置，而是直接把值传过去，表明当前子树的结束点。不过总感觉还是没有 get 到他的点，in 和 stop 变量的含义也是我赋予的，对于整个算法也只是勉强说通，大家有好的想法可以和我交流。
     *
     * 点击下方链接查看
     * StefanPochmann
     * Simple O(n) without map
     * https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/discuss/34543/Simple-O(n)-without-map
     */
    public TreeNode buildTree3(int[] preorder, int[] inorder) {
        return buildTreeHelper(preorder, inorder, (long) Integer.MAX_VALUE + 1);
    }

    int pre = 0;
    int in = 0;

    private TreeNode buildTreeHelper(int[] preorder, int[] inorder, long stop) {
        //到达末尾返回 null
        if (pre == preorder.length) {
            return null;
        }
        //到达停止点返回 null
        //当前停止点已经用了，in 后移
        if (inorder[in] == stop) {
            in++;
            return null;
        }
        int root_val = preorder[pre++];
        TreeNode root = new TreeNode(root_val);
        //左子树的停止点是当前的根节点
        root.left = buildTreeHelper(preorder, inorder, root_val);
        //右子树的停止点是当前树的停止点
        root.right = buildTreeHelper(preorder, inorder, stop);
        return root;
    }

    /**
     * 解法二、迭代 栈
     * 参考 这里，我们可以利用一个栈，用迭代实现。
     * 假设我们要还原的树是下图
     */
    public TreeNode buildTree4(int[] preorder, int[] inorder) {
        if (preorder.length == 0) {
            return null;
        }
        Stack<TreeNode> roots = new Stack<TreeNode>();
        int pre = 0;
        int in = 0;
        //先序遍历第一个值作为根节点
        TreeNode curRoot = new TreeNode(preorder[pre]);
        TreeNode root = curRoot;
        roots.push(curRoot);
        pre++;
        //遍历前序遍历的数组
        while (pre < preorder.length) {
            //出现了当前节点的值和中序遍历数组的值相等，寻找是谁的右子树
            if (curRoot.val == inorder[in]) {
                //每次进行出栈，实现倒着遍历
                while (!roots.isEmpty() && roots.peek().val == inorder[in]) {
                    curRoot = roots.peek();
                    roots.pop();
                    in++;
                }
                //设为当前的右孩子
                curRoot.right = new TreeNode(preorder[pre]);
                //更新 curRoot
                curRoot = curRoot.right;
                roots.push(curRoot);
                pre++;
            } else {
                //否则的话就一直作为左子树
                curRoot.left = new TreeNode(preorder[pre]);
                curRoot = curRoot.left;
                roots.push(curRoot);
                pre++;
            }
        }
        return root;
    }

    /**
     * 77. 组合（微软、亚马逊、谷歌在半年内面试中考过）
     * 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
     * *****************************************************************************************************************
     * 回溯算法 + 剪枝（Java）
     * liweiwei1419
     * https://leetcode-cn.com/problems/combinations/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-ma-/
     */
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (k <= 0 || n < k) {
            return res;
        }
        // 从 1 开始是题目的设定
        Deque<Integer> path = new ArrayDeque<>();
        dfs(n, k, 1, path, res);
        return res;
    }

    private void dfs(int n, int k, int begin, Deque<Integer> path, List<List<Integer>> res) {
        // 递归终止条件是：path 的长度等于 k
        if (path.size() == k) {
            res.add(new ArrayList<>(path));
            return;
        }

        // 遍历可能的搜索起点
        for (int i = begin; i <= n; i++) {
            // 向路径变量里添加一个数
            path.addLast(i);
            System.out.println("递归之前 => " + path);
            // 下一轮搜索，设置的搜索起点要加 1，因为组合数理不允许出现重复的元素
            dfs(n, k, i + 1, path, res);
            // 重点理解这里：深度优先遍历有回头的过程，因此递归之前做了什么，递归之后需要做相同操作的逆向操作
            path.removeLast();
            System.out.println("递归之后 => " + path);
        }
    }

    /**
     * 参考代码 3：
     * 优化：分析搜索起点的上界进行剪枝
     * 我们上面的代码，搜索起点遍历到 n，即：递归函数中有下面的代码片段：
     * // 从当前搜索起点 begin 遍历到 n
     * for (int i = begin; i <= n; i++) {
     *     path.addLast(i);
     *     dfs(n, k, i + 1, path, res);
     *     path.removeLast();
     * }
     *
     * 链接：https://leetcode-cn.com/problems/combinations/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-ma-/
     *
     * 说明：对于回溯算法还比较陌生的朋友，可以参考我的题解 《回溯算法入门级详解 + 练习（持续更新）》。
     * https://leetcode-cn.com/problems/permutations/solution/hui-su-suan-fa-python-dai-ma-java-dai-ma-by-liweiw/
     */
    public List<List<Integer>> combine2(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (k <= 0 || n < k) {
            return res;
        }
        Deque<Integer> path = new ArrayDeque<>();
        dfs2(n, k, 1, path, res);
        return res;
    }

    private void dfs2(int n, int k, int index, Deque<Integer> path, List<List<Integer>> res) {
        if (path.size() == k) {
            res.add(new ArrayList<>(path));
            return;
        }

        // 只有这里 i <= n - (k - path.size()) + 1 与参考代码 1 不同
        for (int i = index; i <= n - (k - path.size()) + 1; i++) {
            path.addLast(i);
            System.out.println("递归之前 => " + path);
            dfs2(n, k, i + 1, path, res);
            path.removeLast();
            System.out.println("递归之后 => " + path);
        }
    }

    /**
     * 参考代码 4：
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/combinations/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-ma-/
     */
    public List<List<Integer>> combine4(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (k <= 0 || n < k) {
            return res;
        }

        // 为了防止底层动态数组扩容，初始化的时候传入最大长度
        Deque<Integer> path = new ArrayDeque<>(k);
        dfs4(1, n, k, path, res);
        return res;
    }

    private void dfs4(int begin, int n, int k, Deque<Integer> path, List<List<Integer>> res) {
        if (k == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        // 基础版本的递归终止条件：if (begin == n + 1) {
        if (begin > n - k + 1) {
            return;
        }
        // 不选当前考虑的数 begin，直接递归到下一层
        dfs4(begin + 1, n, k, path, res);

        // 不选当前考虑的数 begin，递归到下一层的时候 k - 1，这里 k 表示还需要选多少个数
        path.addLast(begin);
        dfs4(begin + 1, n, k - 1, path, res);
        // 深度优先遍历有回头的过程，因此需要撤销选择
        path.removeLast();
    }


    /**
     * 46. 全排列（字节跳动在半年内面试常考）
     * 给定一个 没有重复 数字的序列，返回其所有可能的全排列。
     * *****************************************************************************************************************
     * 示例:
     *
     * 输入: [1,2,3]
     * 输出:
     * [
     *   [1,2,3],
     *   [1,3,2],
     *   [2,1,3],
     *   [2,3,1],
     *   [3,1,2],
     *   [3,2,1]
     * ]
     * *****************************************************************************************************************
     *
     *
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();

        List<Integer> output = new ArrayList<Integer>();
        for (int num : nums) {
            output.add(num);
        }

        int n = nums.length;
        backtrack(n, output, res, 0);
        return res;
    }

    public void backtrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
        // 所有数都填完了
        if (first == n) {
            res.add(new ArrayList<Integer>(output));
        }
        for (int i = first; i < n; i++) {
            // 动态维护数组
            Collections.swap(output, first, i);
            // 继续递归填下一个数
            backtrack(n, output, res, first + 1);
            // 撤销操作
            Collections.swap(output, first, i);
        }
    }

    /**
     * 回溯算法入门级详解 + 练习（持续更新）
     * liweiwei1419
     */
    public List<List<Integer>> permute2(int[] nums) {
        int len = nums.length;
        // 使用一个动态数组保存所有可能的全排列
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        boolean[] used = new boolean[len];
        List<Integer> path = new ArrayList<>();

        dfs(nums, len, 0, path, used, res);
        return res;
    }

    private void dfs(int[] nums, int len, int depth, List<Integer> path, boolean[] used, List<List<Integer>> res) {
        if (depth == len) {
            res.add(path);
            return;
        }

        // 在非叶子结点处，产生不同的分支，这一操作的语义是：在还未选择的数中依次选择一个元素作为下一个位置的元素，这显然得通过一个循环实现。
        for (int i = 0; i < len; i++) {
            if (!used[i]) {
                path.add(nums[i]);
                used[i] = true;

                dfs(nums, len, depth + 1, path, used, res);
                // 注意：下面这两行代码发生 「回溯」，回溯发生在从 深层结点 回到 浅层结点 的过程，代码在形式上和递归之前是对称的
                used[i] = false;
                path.remove(path.size() - 1);
            }
        }
    }

    /**
     * 两种实现+详细图解 46.全排列
     * 解法一
     */
    public List<List<Integer>> permute3(int[] nums) {
        if (nums == null) {
            return new ArrayList<List<Integer>>();
        }
        //最终结果集
        List<List<Integer>> res = new LinkedList<List<Integer>>();
        //将输入的数组放到队列中
        Queue<Integer> queue = new LinkedList<Integer>();
        for (int i : nums) {
            queue.offer(i);
        }
        dfs(res, queue, new LinkedList<Integer>());
        return res;
    }

    private void dfs(List<List<Integer>> res, Queue<Integer> queue, LinkedList<Integer> arr) {
        //如果队列为空，则所有的元素都放入列表(栈)中了，将列表保存到结果集中
        if (queue.isEmpty()) {
            res.add(new LinkedList(arr));
            return;
        }
        //循环次数为队列的长度
        int size = queue.size();
        //从队列中取出第一个元素，放入列表(栈)中，并继续下一层递归
        //等下一层递归返回后，将列表(栈)中的元素弹出，放回到队列中
        for (int i = 0; i < size; ++i) {
            arr.add(queue.poll());
            dfs(res, queue, arr);
            queue.offer(arr.removeLast());
        }
    }

    /**
     * 解法二
     * 我们也可以直接在数组上做交换
     *
     * https://leetcode-cn.com/problems/permutations/solution/liang-chong-shi-xian-xiang-xi-tu-jie-46quan-pai-li/
     */
    public List<List<Integer>> permute4(int[] nums) {
        if (nums == null) {
            return new ArrayList<List<Integer>>();
        }
        List<List<Integer>> res = new LinkedList<List<Integer>>();
        dfs(res, nums, 0);
        return res;
    }

    private void dfs(List<List<Integer>> res, int[] nums, int index) {
        //如果数组下标索引等于nums长度，说明所有元素都选择完了
        if (index == nums.length) {
            List<Integer> list = new ArrayList<Integer>();
            for (int i : nums) {
                list.add(i);
            }
            res.add(list);
            return;
        }
        //第一层递归index为0，第二层递归index为1，以此类推
        //不断交换i和index位置，交换后继续下一层递归，返回后撤销交换
        for (int i = index; i < nums.length; ++i) {
            swap(nums, i, index);
            dfs(res, nums, index + 1);
            swap(nums, i, index);
        }
    }

    private void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    /**
     * 超哥
     * 以交换数组元素的方式做，时空效率均很高。解释：求1—n-1的全排列，过程应是这样的：先将第一个数固定，然后求剩下的n-1个数的全排列。
     * 求n-1个数的全排列的时候，还是一样，将第一个数固定，求n-2个数的全排列。直到剩下一个数，那他的全排列就是自己。那这个固定的数字应该有多种取值，
     * 比如求0,1,2三个数的全排列，固定的第一个数，应有0,1,2三种取值对吧，当固定第一个数，
     * 比如固定了0，那剩下1,2两个数，再固定一个数，这个数有1和2两种取值，有没有发现什么？
     * 我们发现，这个固定的数的取值，不就是将固定的位置的数和剩下的数字不断交换的过程么。
     */
    public List<List<Integer>> permute5(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length == 0)
            return res;
        helper(nums, 0, nums.length, res);
        return res;
    }

    public void helper(int[] nums, int start, int end, List<List<Integer>> res) {
        if (start == end - 1) {
            List<Integer> item = new ArrayList<>();
            for (int num : nums)
                item.add(num);
            res.add(item);
        }
        for (int i = start; i < end; ++i) {
            swap2(nums, start, i);
            helper(nums, start + 1, end, res);
            swap2(nums, start, i);
        }
    }

    public void swap2(int[] nums, int index1, int index2) {
        int tmp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = tmp;
    }

    // ------------------------------------------------ 这 是 一 条 分 割 线 ----------------------------------------------
    // ------------------------------------------------ 这 是 一 条 分 割 线 ----------------------------------------------
    /**
     * A general approach to backtracking questions in Java (Subsets, Permutations, Combination Sum, Palindrome Partioning)
     *
     * This structure might apply to many other backtracking questions, but here I am just going to demonstrate Subsets, Permutations, and Combination Sum.
     *
     * https://leetcode.com/problems/permutations/discuss/18239/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partioning)
     *
     * *****************************************************************************************************************
     *
     * Subsets : https://leetcode.com/problems/subsets/
     *
     * 详情见 Week5.java
     */
    // ------------------------------------------------ 这 是 一 条 分 割 线 ----------------------------------------------
    // ------------------------------------------------ 这 是 一 条 分 割 线 ----------------------------------------------

    /**
     * 47. 全排列 II （亚马逊、字节跳动、Facebook 在半年内面试中考过）
     * 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
     * *****************************************************************************************************************
     * 示例 1：
     *
     * 输入：nums = [1,1,2]
     * 输出：
     * [[1,1,2],
     *  [1,2,1],
     *  [2,1,1]]
     * 示例 2：
     *
     * 输入：nums = [1,2,3]
     * 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
     * *****************************************************************************************************************
     *
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        int len = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        // 排序（升序或者降序都可以），排序是剪枝的前提
        Arrays.sort(nums);

        boolean[] used = new boolean[len];
        // 使用 Deque 是 Java 官方 Stack 类的建议
        Deque<Integer> path = new ArrayDeque<>(len);
        dfs(nums, len, 0, used, path, res);
        return res;
    }

    private void dfs(int[] nums, int len, int depth, boolean[] used, Deque<Integer> path, List<List<Integer>> res) {
        if (depth == len) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < len; ++i) {
            if (used[i]) {
                continue;
            }

            // 剪枝条件：i > 0 是为了保证 nums[i - 1] 有意义
            // 写 !used[i - 1] 是因为 nums[i - 1] 在深度优先遍历的过程中刚刚被撤销选择
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }

            path.addLast(nums[i]);
            used[i] = true;

            dfs(nums, len, depth + 1, used, path, res);
            // 回溯部分的代码，和 dfs 之前的代码是对称的
            used[i] = false;
            path.removeLast();
        }
    }




    public static void main(String[] args) {
        Week3 week3 = new Week3();
        // 计算 x 的 n 次幂函数
        int x = 2;
        int n = -2;
        double pow = week3.myPow(x, n);
        double pow2 = week3.myPow2(x, n);
        double pow3 = week3.myPow3(x, n);
        System.out.println("pow=" + pow + "   pow2=" + pow2 + "   pow3=" + pow3);

        // 子集枚举
        int[] nums_subsets = new int[]{1, 2, 3};
        List<List<Integer>> subsets1 = week3.subsets(nums_subsets);
        List<List<Integer>> subsets2 = Week3.enumerate(nums_subsets);
        // Week3.recursion(nums_subsets, 2,null );
        System.out.println(subsets1 + "      " + subsets2);

        // 括号生成
        System.out.println(week3.generateParenthesis(4));
        System.out.println(week3.generateParenthesis2(4));

        // LCA =  Lowest Common Ancestor of a Binary Tree
        //Solution.TreeNode treeNode = solution.lowestCommonAncestor(root, p, q);


        int n2 = 5;
        int k = 3;
        List<List<Integer>> res = week3.combine(n2, k);
        System.out.println(res);
    }
}
