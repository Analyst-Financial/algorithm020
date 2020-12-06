package algorithm;

import javax.swing.tree.TreeNode;
import java.util.*;

/**
 * @ClassName Week3
 * @Description
 * @Author Administrator
 * @Date 2020/11/30  22:06
 * @Version 1.0
 **/
public class Week3 {

    /**
     * 实现 pow(x, n) ，即计算 x 的 n 次幂函数。
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

// ---------------------------------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------------------------------

    /**
     * 2. 实战题目解析：Pow(x,n)、子集
     * 实战题目
     * Pow(x, n) （Facebook 在半年内面试常考）
     * 子集（Facebook、字节跳动、亚马逊在半年内面试中考过）
     * <p>
     * 方法一：快速幂 + 递归
     */
    public double quickMul(double x, long N) {
        if (N == 0) {
            return 1.0;
        }
        double y = quickMul(x, N / 2);
        return N % 2 == 0 ? y * y : y * y * x;
    }

    /**
     * 实现 pow(x, n) ，即计算 x 的 n 次幂函数。
     */
    public double myPow2(double x, int n) {
        long N = n;
        return N >= 0 ? quickMul(x, N) : 1.0 / quickMul(x, -N);
    }

    /**
     * 方法二：快速幂 + 迭代
     */

    double quickMul3(double x, long N) {
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
        return N >= 0 ? quickMul3(x, N) : 1.0 / quickMul3(x, -N);
    }

    /**
     * 子集（Facebook、字节跳动、亚马逊在半年内面试中考过）
     * 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）说明：解集不能包含重复的子集。
     * 方法一：递归法实现子集枚举
     * [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3], []]
     */
    List<Integer> t = new ArrayList<Integer>();
    List<List<Integer>> ans = new ArrayList<List<Integer>>();

    public List<List<Integer>> subsets(int[] nums) {
        dfs(0, nums);
        return ans;
    }

    public void dfs(int cur, int[] nums) {
        // terminator
        if (cur == nums.length) {
            ans.add(new ArrayList<Integer>(t));
            return;
        }
        t.add(nums[cur]);
        dfs(cur + 1, nums);
        t.remove(t.size() - 1);
        dfs(cur + 1, nums);
    }

    /**
     * 超哥的方法
     * def subsets(self, nums):
     * result = [[]]
     * for num in nums:
     * newsets = []
     * for subset in result:
     * new_subset = subset + [num]
     * newsets.append(new_subset)
     * result.extend(newsets)
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
            ans.add(new ArrayList<Integer>(list));
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
     */
    /**
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
     *
     * 集合中每个元素的选和不选，构成了一个满二叉状态树，比如，左子树是不选，右子树是选，从根节点、到叶子节点的所有路径，构成了所有子集。
     * 可以有前序、中序、后序的不同写法，结果的顺序不一样。本质上，其实是比较完整的中序遍历。
     * 链接：https://leetcode-cn.com/problems/subsets/solution/er-jin-zhi-wei-zhu-ge-mei-ju-dfssan-chong-si-lu-9c/
     */
    /**
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
     * 括号生成 (字节跳动在半年内面试中考过)
     * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
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
     * left 随时可以加,只要别超标
     * right 左个数>右个数
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
     * 实战题目
     * 爬楼梯（阿里巴巴、腾讯、字节跳动在半年内面试常考）
     * 括号生成 (字节跳动在半年内面试中考过)
     * 翻转二叉树 (谷歌、字节跳动、Facebook 在半年内面试中考过)
     * 验证二叉搜索树（亚马逊、微软、Facebook 在半年内面试中考过）
     * 二叉树的最大深度（亚马逊、微软、字节跳动在半年内面试中考过）
     * 二叉树的最小深度（Facebook、字节跳动、谷歌在半年内面试中考过）
     * 二叉树的序列化与反序列化（Facebook、亚马逊在半年内面试常考）
     * 课后作业
     * 二叉树的最近公共祖先（Facebook 在半年内面试常考）
     * 从前序与中序遍历序列构造二叉树（字节跳动、亚马逊、微软在半年内面试中考过）
     * 组合（微软、亚马逊、谷歌在半年内面试中考过）
     * 全排列（字节跳动在半年内面试常考）
     * 全排列 II （亚马逊、字节跳动、Facebook 在半年内面试中考过）
     */


    /**
     * 二叉树的最近公共祖先（Facebook 在半年内面试常考）
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     * https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/solution/er-cha-shu-de-zui-jin-gong-gong-zu-xian-by-leetc-2/
     */
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     * int val;
     * TreeNode left;
     * TreeNode right;
     * TreeNode(int x) { val = x; }
     * }
     */
    // Definition for a binary tree node.
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    private TreeNode treeNode;

    public Week3() {
        this.treeNode = null;
    }

    private boolean dfs(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return false;
        boolean lson = dfs(root.left, p, q);
        boolean rson = dfs(root.right, p, q);
        if ((lson && rson) || ((root.val == p.val || root.val == q.val) && (lson || rson))) {
            treeNode = root;
        }
        return lson || rson || (root.val == p.val || root.val == q.val);
    }

    /**
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
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        this.dfs(root, p, q);
        return this.treeNode;
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
     * 从前序与中序遍历序列构造二叉树（字节跳动、亚马逊、微软在半年内面试中考过）
     * 根据一棵树的前序遍历与中序遍历构造二叉树。
     */
    private int in = 0;
    private int pre = 0;

    public TreeNode buildTree2(int[] preorder, int[] inorder) {
        return build(preorder, inorder, Integer.MIN_VALUE);
    }

    private TreeNode build(int[] preorder, int[] inorder, int stop) {
        if (pre >= preorder.length) return null;
        if (inorder[in] == stop) {
            in++;
            return null;
        }
        TreeNode node = new TreeNode(preorder[pre++]);
        node.left = build(preorder, inorder, node.val);
        node.right = build(preorder, inorder, stop);
        return node;
    }

    /**
     * 方法一：递归
     * 思路
     * <p>
     * 对于任意一颗树而言，前序遍历的形式总是
     * [ 根节点, [左子树的前序遍历结果], [右子树的前序遍历结果] ]
     * 即根节点总是前序遍历中的第一个节点。而中序遍历的形式总是
     * [ [左子树的中序遍历结果], 根节点, [右子树的中序遍历结果] ]
     * 只要我们在中序遍历中定位到根节点，那么我们就可以分别知道左子树和右子树中的节点数目。由于同一颗子树的前序遍历和中序遍历的长度显然是相同的，因此我们就可以对应到前序遍历的结果中，对上述形式中的所有左右括号进行定位。
     * <p>
     * 这样以来，我们就知道了左子树的前序遍历和中序遍历结果，以及右子树的前序遍历和中序遍历结果，我们就可以递归地对构造出左子树和右子树，再将这两颗子树接到根节点的左右位置。
     * <p>
     * 细节
     * <p>
     * 在中序遍历中对根节点进行定位时，一种简单的方法是直接扫描整个中序遍历的结果并找出根节点，但这样做的时间复杂度较高。我们可以考虑使用哈希映射（HashMap）来帮助我们快速地定位根节点。对于哈希映射中的每个键值对，键表示一个元素（节点的值），值表示其在中序遍历中的出现位置。在构造二叉树的过程之前，我们可以对中序遍历的列表进行一遍扫描，就可以构造出这个哈希映射。在此后构造二叉树的过程中，我们就只需要 O(1)O(1) 的时间对根节点进行定位了。
     * 链接：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solution/cong-qian-xu-yu-zhong-xu-bian-li-xu-lie-gou-zao-9/
     */
    private Map<Integer, Integer> indexMap;

    public TreeNode myBuildTree(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
        if (preorder_left > preorder_right) {
            return null;
        }

        // 前序遍历中的第一个节点就是根节点
        int preorder_root = preorder_left;
        // 在中序遍历中定位根节点
        int inorder_root = indexMap.get(preorder[preorder_root]);

        // 先把根节点建立出来
        TreeNode root = new TreeNode(preorder[preorder_root]);
        // 得到左子树中的节点数目
        int size_left_subtree = inorder_root - inorder_left;
        // 递归地构造左子树，并连接到根节点
        // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
        root.left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1);
        // 递归地构造右子树，并连接到根节点
        // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
        root.right = myBuildTree(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right);
        return root;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        // 构造哈希映射，帮助我们快速定位根节点
        indexMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return myBuildTree(preorder, inorder, 0, n - 1, 0, n - 1);
    }

    /**
     * 全排列 II （亚马逊、字节跳动、Facebook 在半年内面试中考过）
     * 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
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


}
