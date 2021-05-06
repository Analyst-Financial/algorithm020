package algorithm;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * @ClassName Tpl
 * @Description 分治代码模板
 * @Author Administrator
 * @Date 2020/12/1  22:23
 * @Version 1.0
 **/
public class Tpl {

    /**
     * DFS 代码模板
     * 递归写法
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> allResults = new ArrayList<>();
        if (root == null) {
            return allResults;
        }
        travel(root, 0, allResults);
        return allResults;
    }


    private void travel(TreeNode root, int level, List<List<Integer>> results) {
        if (results.size() == level) {
            results.add(new ArrayList<>());
        }
        results.get(level).add(root.val);
        if (root.left != null) {
            travel(root.left, level + 1, results);
        }
        if (root.right != null) {
            travel(root.right, level + 1, results);
        }
    }


    /**
     * BFS 代码模板
     */
    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> allResults = new ArrayList<>();
        if (root == null) {
            return allResults;
        }
        Queue<TreeNode> nodes = new LinkedList<>();
        nodes.add(root);
        while (!nodes.isEmpty()) {
            int size = nodes.size();
            List<Integer> results = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = nodes.poll();
                results.add(node.val);
                if (node.left != null) {
                    nodes.add(node.left);
                }
                if (node.right != null) {
                    nodes.add(node.right);
                }
            }
            allResults.add(results);
        }
        return allResults;
    }

    /**
     * 递归代码模版
     * public void recur(int level, int param) {
     *      // terminator
     *      if (level > MAX_LEVEL) {
     *          // process result
     *          return;
     *      }
     *      // process current logic
     *      process(level, param);
     *      // drill down
     *      recur( level: level + 1, newParam);
     *      // restore current status
     * }
     */

    /**
     * 分治代码模板
     *  def divide_conquer(problem, param1, param2, ...):
     *      # recursion terminator
     *      if problem is None:
     *          print_result
     *          return
     *
     *      # prepare data
     *      data = prepare_data(problem)
     *      subproblems = split_problem(problem, data)
     *
     *      # conquer subproblems
     *      subresult1 = self.divide_conquer(subproblems[0], p1, ...)
     *      subresult2 = self.divide_conquer(subproblems[1], p1, ...)
     *      subresult3 = self.divide_conquer(subproblems[2], p1, ...)
     *      …
     *
     *      # process and generate the final result
     *      result = process_result(subresult1, subresult2, subresult3, …)
     *      # revert the current level states
     *
     * 感触
     * 1. 人肉递归低效、很累
     * 2. 找到最近最简方法，将其拆解成可重复解决的问题
     * 3. 数学归纳法思维（抵制人肉递归的诱惑）
     * 本质：寻找重复性 —> 计算机指令集
     */

    /**
     *
     * 动态规划 Dynamic Programming
     * 1. Wiki 定义：
     *    https://en.wikipedia.org/wiki/Dynamic_programming
     * 2.“Simplifying a complicated problem by breaking it down into
     *    simpler sub-problems”
     *    (in a recursive manner)
     * 3.Divide & Conquer + Optimal substructure
     * 分治 + 最优子结构
     * *****************************************************************************************************************
     * 关键点
     * 动态规划 和 递归或者分治 没有根本上的区别（关键看有无最优的子结构）
     * 共性：找到重复子问题
     * 差异性：最优子结构、中途可以淘汰次优解
     * *****************************************************************************************************************
     * 状态转移方程（DP 方程）
     * opt[i , j] = opt[i + 1, j] + opt[i, j + 1]
     * 完整逻辑：
     * if a[i, j] = ‘空地’:
     *    opt[i , j] = opt[i + 1, j] + opt[i, j + 1]
     * else:
     *    opt[i , j] = 0
     * *****************************************************************************************************************
     * 实战例题二
     * 路径计数 Count the paths
     * 动态规划关键点
     * 1. 最优子结构 opt[n] = best_of(opt[n-1], opt[n-2], …)
     * 2. 储存中间状态：opt[i]
     * 3. 递推公式（美其名曰：状态转移方程或者 DP 方程）
     * Fib: opt[i] = opt[n-1] + opt[n-2]
     * 二维路径：opt[i,j] = opt[i+1][j] + opt[i][j+1] (且判断a[i,j]是否空地）
     * *****************************************************************************************************************
     * 实战例题三
     * 最长公共子序列
     * 子问题
     * • S1 = “ABAZDC”
     *   S2 = “BACBAD”
     * • If S1[-1] != S2[-1]: LCS[s1, s2] = Max(LCS[s1-1, s2], LCS[s1, s2-1])
     *   LCS[s1, s2] = Max(LCS[s1-1, s2], LCS[s1, s2-1], LCS[s1-1, s2-1])
     * • If S1[-1] == S2[-1]: LCS[s1, s2] = LCS[s1-1, s2-1] + 1
     *   LCS[s1, s2] = Max(LCS[s1-1, s2], LCS[s1, s2-1], LCS[s1-1, s2-1], LCS[s1-1][s2-1] + 1)
     * *****************************************************************************************************************
     * DP 方程
     * • If S1[-1] != S2[-1]: LCS[s1, s2] = Max(LCS[s1-1, s2], LCS[s1, s2-1])
     * • If S1[-1] == S2[-1]: LCS[s1, s2] = LCS[s1-1, s2-1] + 1
     * 动态规划小结
     * 1. 打破自己的思维惯性，形成机器思维
     * 2. 理解复杂逻辑的关键
     * 3. 也是职业进阶的要点要领
     * MIT algorithm course
     * B 站搜索： mit 动态规划
     * https://www.bilibili.com/video/av53233912?from=search&seid=2847395688604491997
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

}
