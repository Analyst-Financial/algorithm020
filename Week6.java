package algorithm;

import java.util.*;

/**
 * @ClassName Week6
 * @Description
 * @Author Administrator
 * @Date 2020/12/20  21:30
 * @Version 1.0
 **/
public class Week6 {

    /**
     * 本周作业及下周预习
     * *****************************************************************************************************************
     * 本周作业
     * 中等
     * 最小路径和（亚马逊、高盛集团、谷歌在半年内面试中考过）
     * 解码方法（亚马逊、Facebook、字节跳动在半年内面试中考过）
     * 最大正方形（华为、谷歌、字节跳动在半年内面试中考过）
     * 任务调度器（Facebook 在半年内面试中常考）
     * 回文子串（Facebook、苹果、字节跳动在半年内面试中考过）
     * 困难
     * 最长有效括号（字节跳动、亚马逊、微软在半年内面试中考过）
     * 编辑距离（字节跳动、亚马逊、谷歌在半年内面试中考过）
     * 矩形区域不超过 K 的最大数值和（谷歌在半年内面试中考过）
     * 青蛙过河（亚马逊、苹果、字节跳动在半年内面试中考过）
     * 分割数组的最大值（谷歌、亚马逊、Facebook 在半年内面试中考过）
     * 学生出勤记录 II （谷歌在半年内面试中考过）
     * 最小覆盖子串（Facebook 在半年内面试中常考）
     * 戳气球（亚马逊在半年内面试中考过）
     * 下周预习
     * 预习题目：
     * 实现 Trie (前缀树)
     * 单词搜索 II
     * 岛屿数量
     * 有效的数独
     * N 皇后
     * 单词接龙
     * 二进制矩阵中的最短路径
     */

    /**
     * 62. 不同路径
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     * 问总共有多少条不同的路径？
     * *****************************************************************************************************************
     * 超哥方法1
     */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; ++i) dp[i][0] = 1;
        for (int j = 0; j < n; ++j) dp[0][j] = 1;
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * 超哥方法2
     * 从上往下 && 从左往右 循环
     */
    public int uniquePaths2(int m, int n) {
        int[][] dp = new int[m][n]; //从<start>走到(i,j)的不同路径数
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0)
                    dp[i][j] = 1;
                else
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * 超哥方法3
     * 从下往上 && 从右往左 循环
     */
    public int uniquePaths3(int m, int n) {
        int[][] dp = new int[m][n]; //从(i,j)走到end的不同路径数
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if (i == m - 1 || j == n - 1)
                    dp[i][j] = 1;
                else
                    dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
            }
        }
        return dp[0][0];
    }

    /**
     * 方法4
     */
    public int uniquePaths4(int m, int n) {
        int[] cur = new int[n];
        Arrays.fill(cur, 1);
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                cur[j] += cur[j - 1];
            }
        }
        return cur[n - 1];
    }

    /**
     * 63. 不同路径 II （谷歌、美团、微软在半年内面试中考过）
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
     * 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
     * 网格中的障碍物和空位置分别用 1 和 0 来表示。
     * 输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
     * 输出：2
     * *****************************************************************************************************************
     * 超哥的方法:
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[] dp = new int[n + 1];
        dp[1] = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 1; j <= n; j++) {
                if (obstacleGrid[i][j - 1] == 1) { //障碍物
                    dp[j] = 0;
                } else {
                    dp[j] = dp[j] + dp[j - 1]; //dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[n];
    }
    public int uniquePathsWithObstacles2(int[][] obstacleGrid) {
        int width = obstacleGrid[0].length;
        int[] dp = new int[width];
        dp[0] = 1;
        for (int[] row : obstacleGrid) {
            for (int j = 0; j < width; j++) {
                if (row[j] == 1)
                    dp[j] = 0;
                else if (j > 0)
                    dp[j] += dp[j - 1];
            }
        }
        return dp[width - 1];
    }

    /**
     * 1143. 最长公共子序列
     * 给定两个字符串text1 和 text2，返回这两个字符串的最长公共子序列的长度。
     * 一个字符串的子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
     * 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。
     * 若这两个字符串没有公共子序列，则返回 0。
     * 输入：text1 = "abcde", text2 = "ace"
     * 输出：3
     * https://leetcode.com/problems/longest-common-subsequence/discuss/351689/JavaPython-3-Two-DP-codes-of-O(mn)-and-O(min(m-n))-spaces-w-picture-and-analysis
     * 这边加一的目的是为了让索引为0的行和列表示空串，不需要额外限制条件去确保index是否是有效的
     */
    public int longestCommonSubsequence(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        for (int i = 0; i < s1.length(); ++i)
            for (int j = 0; j < s2.length(); ++j)
                if (s1.charAt(i) == s2.charAt(j)) dp[i + 1][j + 1] = 1 + dp[i][j];
                else dp[i + 1][j + 1] = Math.max(dp[i][j + 1], dp[i + 1][j]);
        return dp[s1.length()][s2.length()];
    }

    /**
     * 5 easy steps to DP
     * ① define subproblems
     * ② guess (part of solution)
     * ③ relate subproblem solutions
     * ④ recurse & memorize OR build DP table bottom-up
     * ⑤ solve original problem
     * MIT 动态规划课程最短路径算法
     * https://www.bilibili.com/video/av53233912?from=search&seid=2847395688604491997
     */

    /**
     * 例如：s1="abcde"　　s2= "ace"，求两个字符串的公共子序列，答案是“ace”
     * 1.　S{s1,s2,s3....si} T{t1,t2,t3,t4....tj}
     * 2.　子问题划分
     * (1) 如果S的最后一位等于T的最后一位，则最大子序列就是{s1,s2,s3...si-1}和{t1,t2,t3...tj-1}的最大子序列+1
     * (2) 如果S的最后一位不等于T的最后一位，那么最大子序列就是
     * ① {s1,s2,s3..si}和 {t1,t2,t3...tj-1} 最大子序列
     * ② {s1,s2,s3...si-1}和{t1,t2,t3....tj} 最大子序列
     * 以上两个自序列的最大值
     * 3.　边界
     * 只剩下{s1}和{t1}，如果相等就返回1，不等就返回0
     * 4.　使用一个表格来存储dp的结果
     * 如果 S[i] == T[j] 则dp[i][j] = dp[i-1][j-1] + 1
     * 否则dp[i][j] = max(dp[i][j-1],dp[i-1][j])
     * *****************************************************************************************************************
     * 链接：https://leetcode-cn.com/problems/longest-common-subsequence/solution/chao-xiang-xi-dong-tai-gui-hua-jie-fa-by-shi-wei-h/
     */
    public int longestCommonSubsequence2(String text1, String text2) {
        char[] s1 = text1.toCharArray();
        char[] s2 = text2.toCharArray();
        int[][] dp = new int[s1.length + 1][s2.length + 1];

        for (int i = 1; i < s1.length + 1; i++) {
            for (int j = 1; j < s2.length + 1; j++) {
                //如果末端相同
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    //如果末端不同
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[s1.length][s2.length];
    }

    /**
     * https://leetcode.com/problems/longest-common-subsequence/discuss/351689/JavaPython-3-Two-DP-codes-of-O(mn)-and-O(min(m-n))-spaces-w-picture-and-analysis
     */
    public int longestCommonSubsequence3(String s1, String s2) {
        int m = s1.length(), n = s2.length();
        if (m < n) return longestCommonSubsequence(s2, s1);
        int[][] dp = new int[2][n + 1];
        for (int i = 0, k = 1; i < m; ++i, k ^= 1)
            for (int j = 0; j < n; ++j)
                if (s1.charAt(i) == s2.charAt(j)) dp[k][j + 1] = 1 + dp[k ^ 1][j];
                else dp[k][j + 1] = Math.max(dp[k ^ 1][j + 1], dp[k][j]);
        return dp[m % 2][n];
    }

    /**
     * https://leetcode.com/problems/longest-common-subsequence/discuss/351689/JavaPython-3-Two-DP-codes-of-O(mn)-and-O(min(m-n))-spaces-w-picture-and-analysis
     */
    public int longestCommonSubsequence4(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        if (m < n) {
            return longestCommonSubsequence(text2, text1);
        }
        int[] dp = new int[n + 1];
        for (int i = 0; i < text1.length(); ++i) {
            for (int j = 0, prevRow = 0, prevRowPrevCol = 0; j < text2.length(); ++j) {
                prevRowPrevCol = prevRow;
                prevRow = dp[j + 1];
                dp[j + 1] = text1.charAt(i) == text2.charAt(j) ? prevRowPrevCol + 1 : Math.max(dp[j], prevRow);
            }
        }
        return dp[n];
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
     * 64. 最小路径和（亚马逊、高盛集团、谷歌在半年内面试中考过）
     * 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     * 说明：每次只能向下或者向右移动一步。
     * 输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
     * 输出：7
     * 解释：因为路径 1→3→1→1→1 的总和最小。
     * https://leetcode-cn.com/problems/minimum-path-sum/solution/zui-xiao-lu-jing-he-dong-tai-gui-hua-gui-fan-liu-c/
     * *****************************************************************************************************************
     * 第5周 第12课 | 动态规划（一）
     * 2. DP例题解析：Fibonacci数列、路径计数
     * 所谓最优子结构：推导出的第n步的值，是前面几个值的最佳值的简单累加，或者最大值，或者最小值。
     * 状态转移方程(DP方程)
     * opt[i,j] = opt[i+1,j] + opt[i,j+1]
     * 完整逻辑:
     * if a[i,j] ='空地' :
     *    opt[i,j] = opt[i+1,j] + opt[i,j+1]
     * else :
     *    opt[i,j] = 0
     * 动态规划关键点
     */
    public int minPathSum(int[][] grid) {
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (i == 0 && j == 0) continue;
                else if (i == 0) grid[i][j] = grid[i][j - 1] + grid[i][j];
                else if (j == 0) grid[i][j] = grid[i - 1][j] + grid[i][j];
                else grid[i][j] = Math.min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j];
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }

    /**
     * 64. Minimum Path Sum
     */
    public int minPathSum2(int[][] grid) {
        int row = grid.length - 1;
        int col = grid[0].length - 1;
        int[][] dp = new int[row + 1][col + 1];
        return minPathSum(grid, row, col, dp);
    }

    public int minPathSum(int[][] grid, int row, int col, int[][] dp) {
        if (row == 0 && col == 0) return grid[row][col];
        if (dp[row][col] != 0) return dp[row][col];
        if (row != 0 && col == 0) return dp[row][col] = grid[row][col] + minPathSum(grid, row - 1, col, dp);
        if (row == 0 && col != 0) return dp[row][col] = grid[row][col] + minPathSum(grid, row, col - 1, dp);
        return dp[row][col] = grid[row][col] + Math.min(minPathSum(grid, row - 1, col, dp), minPathSum(grid, row, col - 1, dp));
    }

    /**
     * I had been struggling for recursive and DP problems before. But now I feeling like I am getting better.
     * This is a simple explanations for those who are still not that good at these just like me.
     * 1.Recursion:
     * So basically let's begin with recursion because it is easier to understand and code. When we think about this problem, we could use a top down approach. To get a path, we need to travel from grid[0][0] to grid[row - 1][col - 1]. So let's set grid[0][0] as the basic case. This is when we jump out of recursion. On the other hand, grid[row - 1][col - 1] would be the starting point. We write a helper function to do the recursion work. At the starting point, this function returns (value of the end cell + value of the cell that has the less one). But we need to consider that things could happen that we reached the first row or column and we gotta make sure that we stay within the array index limit.
     * At last, when we reach grid[0][0], we are done!
     * 2.Dynamic Programming:
     * Now, let's upgrade this algorithm from recursion to DP since we don't wanna get stackoverflow for large inputs.In fact, there is nothing fancy about DP.
     * It is simply that we store or cache the results of every single calculation so that we don't need to calculate the same thing again and again.
     * The whole idea is almost the same. We just involve an array to store the values. Now let's see the code:
     */
    public int minPathSum3(int[][] grid) {
        int height = grid.length;
        int width = grid[0].length;
        return min(grid, height - 1, width - 1);
    }

    public int min(int[][] grid, int row, int col) {
        if (row == 0 && col == 0) return grid[row][col]; // this is the exit of the recursion
        if (row == 0)
            return grid[row][col] + min(grid, row, col - 1); /** when we reach the first row, we could only move horizontally.*/
        if (col == 0)
            return grid[row][col] + min(grid, row - 1, col); /** when we reach the first column, we could only move vertically.*/
        return grid[row][col] + Math.min(min(grid, row - 1, col), min(grid, row, col - 1)); /** we want the min sum path so we pick the cell with the less value */

    }

    public static int minPathSum4(int[][] grid) {
        int height = grid.length;
        int width = grid[0].length;
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                if (row == 0 && col == 0) grid[row][col] = grid[row][col];
                else if (row == 0 && col != 0) grid[row][col] = grid[row][col] + grid[row][col - 1];
                else if (col == 0 && row != 0) grid[row][col] = grid[row][col] + grid[row - 1][col];
                else grid[row][col] = grid[row][col] + Math.min(grid[row - 1][col], grid[row][col - 1]);
            }
        }
        return grid[height - 1][width - 1];
    }

    /**
     * 方法一：动态规划
     * 由于路径的方向只能是向下或向右，因此网格的第一行的每个元素只能从左上角元素开始向右移动到达，网格的第一列的每个元素只能从左上角元素开始向下移动到达，此时的路径是唯一的，因此每个元素对应的最小路径和即为对应的路径上的数字总和。
     * 对于不在第一行和第一列的元素，可以从其上方相邻元素向下移动一步到达，或者从其左方相邻元素向右移动一步到达，元素对应的最小路径和等于其上方相邻元素与其左方相邻元素两者对应的最小路径和中的最小值加上当前元素的值。由于每个元素对应的最小路径和与其相邻元素对应的最小路径和有关，因此可以使用动态规划求解。
     * 创建二维数组 dp，与原始网格的大小相同，dp[i][j] 表示从左上角出发到 (i,j) 位置的最小路径和。显然，dp[0][0]=grid[0][0]。对于 dp 中的其余元素，通过以下状态转移方程计算元素值。
     * 当 i>0 且 j=0 时，dp[i][0]=dp[i−1][0]+grid[i][0]。
     * 当 i=0 且 j>0 时，dp[0][j]=dp[0][j−1]+grid[0][j]。
     * 当 i>0 且 j>0 时，dp[i][j]=min(dp[i−1][j],dp[i][j−1])+grid[i][j]。
     * 最后得到 dp[m−1][n−1] 的值即为从网格左上角到网格右下角的最小路径和。
     * *****************************************************************************************************************
     * 复杂度分析
     * 时间复杂度：O(mn)，其中 m 和 n 分别是网格的行数和列数。需要对整个网格遍历一次，计算 dp 的每个元素的值。
     * 空间复杂度：O(mn)，其中 m 和 n 分别是网格的行数和列数。创建一个二维数组 dp，和网格大小相同。
     * 空间复杂度可以优化，例如每次只存储上一行的 dp 值，则可以将空间复杂度优化到 O(n)。
     * *****************************************************************************************************************
     * 链接：https://leetcode-cn.com/problems/minimum-path-sum/solution/zui-xiao-lu-jing-he-by-leetcode-solution/
     */
    public int minPathSum5(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        int rows = grid.length, columns = grid[0].length;
        int[][] dp = new int[rows][columns];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < rows; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < columns; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < columns; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[rows - 1][columns - 1];
    }

    /**
     * 91. 解码方法（亚马逊、Facebook、字节跳动在半年内面试中考过）
     * 一条包含字母 A-Z 的消息通过以下方式进行了编码：
     * 'A' -> 1
     * 'B' -> 2
     * ...
     * 'Z' -> 26
     * 给定一个只包含数字的非空字符串，请计算解码方法的总数。
     * 题目数据保证答案肯定是一个 32 位的整数。
     * *****************************************************************************************************************
     * I optimize this answer by removing convert Char to Integer, and beats 98% solution:
     */
    public int numDecodings2(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int[] dp = new int[s.length()];
        dp[0] = s.charAt(0) == '0' ? 0 : 1;
        for (int i = 1; i < s.length(); i++) {
            int cur = s.charAt(i) - '0';
            int pre = (s.charAt(i - 1) - '0') * 10 + cur;
            if (cur != 0) {
                dp[i] += dp[i - 1];
            }
            if (pre >= 10 && pre <= 26) {
                dp[i] += i >= 2 ? dp[i - 2] : 1;
            }
        }
        return dp[s.length() - 1];
    }

    /**
     * https://leetcode.com/problems/decode-ways/discuss/30358/Java-clean-DP-solution-with-explanation
     * *****************************************************************************************************************
     * Java clean DP solution with explanation
     * I used a dp array of size n + 1 to save subproblem solutions.
     * dp[0] means an empty string will have one way to decode, dp[1] means the way to decode a string of size 1.
     * I then check one digit and two digit combination and save the results along the way.
     * In the end, dp[n] will be the end result.
     * *****************************************************************************************************************
     * Similar questions:
     * 62. Unique Paths
     * 70. Climbing Stairs
     * 509. Fibonacci Number
     * https://leetcode.com/problems/unique-paths/
     * https://leetcode.com/problems/climbing-stairs/
     * https://leetcode.com/problems/fibonacci-number/
     * *****************************************************************************************************************
     */
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) != '0' ? 1 : 0;
        for (int i = 2; i <= n; i++) {
            int first = Integer.parseInt(s.substring(i - 1, i));
            int second = Integer.parseInt(s.substring(i - 2, i));
            if (first >= 1 && first <= 9) {
                dp[i] += dp[i - 1];
            }
            if (second >= 10 && second <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }

    /**
     * 221. 最大正方形（华为、谷歌、字节跳动在半年内面试中考过）
     * 在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。
     * 输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
     * 输出：4
     * 方法二：动态规划
     * 方法一虽然直观，但是时间复杂度太高，有没有办法降低时间复杂度呢？
     * 可以使用动态规划降低时间复杂度。我们用 dp(i, j) 表示以 (i, j) 为右下角，且只包含 1 的正方形的边长最大值。如果我们能计算出所有 dp(i, j) 的值，那么其中的最大值即为矩阵中只包含 1 的正方形的边长最大值，其平方即为最大正方形的面积。
     * 那么如何计算 dp 中的每个元素值呢？对于每个位置 (i, j)，检查在矩阵中该位置的值：
     * 如果该位置的值是 0，则 dp(i, j) = 0，因为当前位置不可能在由 1 组成的正方形中；
     * 如果该位置的值是 1，则 dp(i, j) 的值由其上方、左方和左上方的三个相邻位置的 dp 值决定。具体而言，当前位置的元素值等于三个相邻位置的元素中的最小值加 1，状态转移方程如下：
     * dp(i, j)=min(dp(i−1, j), dp(i−1, j−1), dp(i, j−1))+1
     * 如果读者对这个状态转移方程感到不解，可以参考 1277. 统计全为 1 的正方形子矩阵的官方题解，其中给出了详细的证明。
     * 此外，还需要考虑边界条件。如果 i 和 j 中至少有一个为 0，则以位置 (i, j) 为右下角的最大正方形的边长只能是 1，因此 dp(i, j) = 1。
     * 链接：https://leetcode-cn.com/problems/maximal-square/solution/zui-da-zheng-fang-xing-by-leetcode-solution/
     */
    public int maximalSquare(char[][] matrix) {
        int maxSide = 0;
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return maxSide;
        }
        int rows = matrix.length, columns = matrix[0].length;
        int[][] dp = new int[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                    maxSide = Math.max(maxSide, dp[i][j]);
                }
            }
        }
        return maxSide * maxSide;
    }

    /**
     *
     * *****************************************************************************************************************
     * 理解 min(上, 左, 左上) + 1
     * 如题，在其他动态规划方法的题解中，大都会涉及到下列形式的代码：
     * // 伪代码
     * if (matrix(i - 1, j - 1) == '1') {
     *     dp(i, j) = min(dp(i - 1, j), dp(i, j - 1), dp(i - 1, j - 1)) + 1;
     * }
     * 其中，dp(i, j) 是以 matrix(i - 1, j - 1) 为 右下角 的正方形的最大边长。(感谢 @liweiwei1419 提出补充)
     * 等同于：dp(i + 1, j + 1) 是以 matrix(i, j) 为右下角的正方形的最大边长
     * 翻译成中文
     * 若某格子值为 1，则以此为右下角的正方形的、最大边长为：上面的正方形、左面的正方形或左上的正方形中，最小的那个，再加上此格。
     * 先来阐述简单共识
     * 若形成正方形（非单 1），以当前为右下角的视角看，则需要：当前格、上、左、左上都是 1
     * 可以换个角度：当前格、上、左、左上都不能受 0 的限制，才能成为正方形
     * 图解如下链接所示
     * https://leetcode-cn.com/problems/maximal-square/solution/li-jie-san-zhe-qu-zui-xiao-1-by-lzhlyle/
     * 上面详解了 三者取最小 的含义：
     * 图 1：受限于左上的 0
     * 图 2：受限于上边的 0
     * 图 3：受限于左边的 0
     * 数字表示：以此为正方形右下角的最大边长
     * 黄色表示：格子 ? 作为右下角的正方形区域
     * 就像 木桶的短板理论 那样——附近的最小边长，才与 ? 的最长边长有关。
     * 此时已可得到递推公式
     * // 伪代码
     * if (grid[i - 1][j - 1] == '1') {
     *     dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1;
     * }
     * *****************************************************************************************************************
     * 从感性理解，到代码实现
     * 从上述图解中，我们似乎得到的只是「动态规划 推进 的过程」，即「如何从前面的 dp 推出后面的 dp」，甚至还只是感性理解
     * 距离代码我们还缺：dp 具体定义如何，数组多大，初值如何，如何与题目要求的面积相关
     * dp 具体定义：dp[i + 1][j + 1] 表示 「以第 i 行、第 j 列为右下角的正方形的最大边长」
     * 为何不是 dp[i][j]
     * 回到图解中，任何一个正方形，我们都「依赖」当前格 左、上、左上三个方格的情况
     * 但第一行的上层已经没有格子，第一列左边已经没有格子，需要做特殊 if 判断来处理
     * 为了代码简洁，我们 假设补充 了多一行全 '0'、多一列全 '0'
     * 如下链接所示
     * 链接：https://leetcode-cn.com/problems/maximal-square/solution/li-jie-san-zhe-qu-zui-xiao-1-by-lzhlyle/
     * 此时 dp 数组的大小也明确为 new dp[height + 1][width + 1]
     * 初始值就是将第一列 dp[row][0] 、第一行 dp[0][col] 都赋为 0，相当于已经计算了所有的第一行、第一列的 dp 值
     * 题目要求面积。根据 「面积 = 边长 x 边长」可知，我们只需求出 最大边长 即可
     * 定义 maxSide 表示最长边长，每次得出一个 dp，就 maxSide = max(maxSide, dp);
     * 最终返回 return maxSide * maxSide;
     * 参考代码
     * 时间复杂度 O(height * width)
     * 空间复杂度 O(height * width)
     * *****************************************************************************************************************
     * 链接：https://leetcode-cn.com/problems/maximal-square/solution/li-jie-san-zhe-qu-zui-xiao-1-by-lzhlyle/
     */
    public int maximalSquare2(char[][] matrix) {
        // base condition
        if (matrix == null || matrix.length < 1 || matrix[0].length < 1) return 0;

        int height = matrix.length;
        int width = matrix[0].length;
        int maxSide = 0;

        // 相当于已经预处理新增第一行、第一列均为0
        int[][] dp = new int[height + 1][width + 1];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                if (matrix[row][col] == '1') {
                    dp[row + 1][col + 1] = Math.min(Math.min(dp[row + 1][col], dp[row][col + 1]), dp[row][col]) + 1;
                    maxSide = Math.max(maxSide, dp[row + 1][col + 1]);
                }
            }
        }
        return maxSide * maxSide;
    }

    /**
     * 优化空间
     * 为了避免到边的判断处理，在最左侧加上一列 dp[i][0] = 0 ，在左上边加上一行 dp[0][j] = 0 ，这才有了官方题解中所谓的 matrix[i - 1][j - 1] == '1' 与 dp[i][j] ，其实都是指可对应上的"当前格子"
     * 其实只需关注"当前格子的周边"，故可二维降一维优化
     * 增加 northwest 西北角解决"左上角"的问题，感谢 @less 指出之前缺漏的 遍历每行时，还原回辅助的原值0 的问题 northwest = 0;
     * *****************************************************************************************************************
     * 时间复杂度 O(height * width)
     * 空间复杂度 O(width)
     * 链接：https://leetcode-cn.com/problems/maximal-square/solution/li-jie-san-zhe-qu-zui-xiao-1-by-lzhlyle/
     */
    public int maximalSquare3(char[][] matrix) {
        if (matrix == null || matrix.length < 1 || matrix[0].length < 1) return 0;

        int height = matrix.length;
        int width = matrix[0].length;
        int maxSide = 0;

        int[] dp = new int[width + 1];

        for (char[] chars : matrix) {
            int northwest = 0; // 个人建议放在这里声明，而非循环体外
            for (int col = 0; col < width; col++) {
                int nextNorthwest = dp[col + 1];
                if (chars[col] == '1') {
                    dp[col + 1] = Math.min(Math.min(dp[col], dp[col + 1]), northwest) + 1;
                    maxSide = Math.max(maxSide, dp[col + 1]);
                } else dp[col + 1] = 0;
                northwest = nextNorthwest;
            }
        }
        return maxSide * maxSide;
    }

    /**
     * https://leetcode.com/problems/maximal-square/discuss/61935/6-lines-Visual-Explanation-O(mn)
     * class Solution:
     *     def maximalSquare(self, A):
     *         for i in range(len(A)):
     *             for j in range(len(A[i])):
     *                 A[i][j] = int(A[i][j])
     *                 if A[i][j] and i and j:
     *                     A[i][j] = min(A[i-1][j], A[i-1][j-1], A[i][j-1]) + 1
     *         return len(A) and max(map(max, A)) ** 2
     */

    /**
     * 621. 任务调度器（Facebook 在半年内面试中常考）
     * 给你一个用字符数组 tasks 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。
     * 然而，两个 相同种类 的任务之间必须有长度为整数 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。
     * 你需要计算完成所有任务所需要的 最短时间 。
     * *****************************************************************************************************************
     * 示例 1：
     * 输入：tasks = ["A","A","A","B","B","B"], n = 2
     * 输出：8
     * 解释：A -> B -> (待命) -> A -> B -> (待命) -> A -> B
     *      在本示例中，两个相同类型任务之间必须间隔长度为 n = 2 的冷却时间，而执行一个任务只需要一个单位时间，所以中间出现了（待命）状态。
     * *****************************************************************************************************************
     * 示例 2：
     * 输入：tasks = ["A","A","A","B","B","B"], n = 0
     * 输出：6
     * 解释：在这种情况下，任何大小为 6 的排列都可以满足要求，因为 n = 0
     * ["A","A","A","B","B","B"]
     * ["A","B","A","B","A","B"]
     * ["B","B","B","A","A","A"]
     * ...
     * 诸如此类
     * *****************************************************************************************************************
     * 示例 3：
     * 输入：tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
     * 输出：16
     * 解释：一种可能的解决方案是：
     *      A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> (待命) -> (待命) -> A -> (待命) -> (待命) -> A
     * *****************************************************************************************************************
     * 方法(贪心算法)
     * 容易想到的一种贪心策略为：先安排出现次数最多的任务，让这个任务两次执行的时间间隔正好为n。再在这个时间间隔内填充其他的任务。
     * 例如：tasks = ["A","A","A","B","B","B"], n = 2
     * 我们先安排出现次数最多的任务"A",并且让两次执行"A"的时间间隔为2。在这个时间间隔内，我们用其他任务类型去填充，又因为其他任务类型只有"B"一个，不够填充2的时间间隔，因此额外需要一个冷却时间间隔。具体安排如下图所示：
     * 链接：https://leetcode-cn.com/problems/task-scheduler/solution/jian-ming-yi-dong-de-javajie-da-by-lan-s-jfl9/
     * 其中，maxTimes为出现次数最多的那个任务出现的次数。maxCount为一共有多少个任务和出现最多的那个任务出现次数一样。
     * 图中一共占用的方格即为完成所有任务需要的时间，即：
     * (maxTimes - 1)*(n + 1) + maxCount
     * 此外，如果任务种类很多，在安排时无需冷却时间，只需要在一个任务的两次出现间填充其他任务，然后从左到右从上到下依次执行即可，由于每一个任务占用一个时间单位，我们又正正好好地使用了tasks中的所有任务，而且我们只使用tasks中的任务来占用方格（没用冷却时间）。因此这种情况下，所需要的时间即为tasks的长度。
     * 由于这种情况时再用上述公式计算会得到一个不正确且偏小的结果，因此，我们只需把公式计算的结果和tasks的长度取最大即为最终结果。
     * 链接：https://leetcode-cn.com/problems/task-scheduler/solution/jian-ming-yi-dong-de-javajie-da-by-lan-s-jfl9/
     */
    public int leastInterval(char[] tasks, int n) {
        int[] buckets = new int[26];
        for (int i = 0; i < tasks.length; i++) {
            buckets[tasks[i] - 'A']++;
        }
        Arrays.sort(buckets);
        int maxTimes = buckets[25];
        int maxCount = 1;
        for (int i = 25; i >= 1; i--) {
            if (buckets[i] == buckets[i - 1])
                maxCount++;
            else
                break;
        }
        int res = (maxTimes - 1) * (n + 1) + maxCount;
        return Math.max(res, tasks.length);
    }

    /**
     * (c[25] - 1) * (n + 1) + 25 - i  is frame size
     * when inserting chars, the frame might be "burst", then tasks.length takes precedence
     * when 25 - i > n, the frame is already full at construction, the following is still valid.
     * *****************************************************************************************************************
     * https://leetcode.com/problems/task-scheduler/discuss/104496/concise-Java-Solution-O(N)-time-O(26)-space
     */
    public int leastInterval2(char[] tasks, int n) {

        int[] c = new int[26];
        for (char t : tasks) {
            c[t - 'A']++;
            System.out.print("  "+t);
            System.out.print(t - 'A'+"  ");
        }
        Arrays.sort(c);
        int i = 25;
        while (i >= 0 && c[i] == c[25]) i--;

        return Math.max(tasks.length, (c[25] - 1) * (n + 1) + 25 - i);
    }


    /**
     * 647. 回文子串（Facebook、苹果、字节跳动在半年内面试中考过）
     * 给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。
     * 具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
     * *****************************************************************************************************************
     * 示例 1：
     * 输入："abc"
     * 输出：3
     * 解释：三个回文子串: "a", "b", "c"
     * *****************************************************************************************************************
     * 示例 2：
     * 输入："aaa"
     * 输出：6
     * 解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
     * *****************************************************************************************************************
     * 647. 回文子串 动态规划方式求解
     * 解题思路
     * 这道题基本和5. 最长回文子串思路是一样的，同样使用动态规划的方法。这里需要找的是最长回文子串，首先第一步，我们需要定义dp数组的含义，定义二维布尔数组dp[i][j]dp[i][j]数组表示：
     * 字符串s[i⋯j]是否为回文子串，如果是，dp[i][j] = true，如果不是，dp[i][j] = false。
     * 我们看如下例子：
     * 链接：https://leetcode-cn.com/problems/palindromic-substrings/solution/647-hui-wen-zi-chuan-dong-tai-gui-hua-fang-shi-qiu/
     * 如何我们现在已经知道了dp[i+1][j-1]了，那我们如何计算dp[i][j]呢？通过观察，我们发现：
     * 如果s[i] == s[j]那么说明只要dp[i+1][j-1]是回文子串，那么是dp[i][j]也就是回文子串
     * 如果s[i] != s[j]那么说明dp[i][j]必定不是回文子串。
     * if(s.charAt(i) == s.charAt(j)){
     *     dp[i][j] = dp[i+1][j-1]
     * }else{
     *     dp[i][j] = false;
     * }
     * 接下来我们需要考虑base case，这里显而易见，当只有一个字母的时候肯定是回文子串，所以初始化的dp表应该如下图所示。
     * 链接：https://leetcode-cn.com/problems/palindromic-substrings/solution/647-hui-wen-zi-chuan-dong-tai-gui-hua-fang-shi-qiu/
     * 遍历的方式呢我们可以按照右下角开始遍历。从右下角遍历的原因是因为(i, j) 位置的值依赖于（i+1,j-1）
     * for(int i = s.length()-1; i>=0; i--){
     *     for(int j = i+1; j<s.length(); j++){
     *         if(s.charAt(i) == s.charAt(j)){
     *             dp[i][j] = dp[i+1][j-1]
     *         }
     *         else{
     *             dp[i][j] = false;
     *         }
     *     }
     * }
     * 这样就基本完成了这道题目。但是这样会有一种情况通过不了例如给的例子中的“cbbd”
     * 链接：https://leetcode-cn.com/problems/palindromic-substrings/solution/647-hui-wen-zi-chuan-dong-tai-gui-hua-fang-shi-qiu/
     * 这道题中的回文子串应该为**“bb”**但是我们的dp表中并没有计算出来，这是因为当计算dp[i][j]dp[i][j]的时候，中间已经没有dp[i+1][j-1]dp[i+1][j−1]了，这是我们在base case中没有考虑到的。
     * 由于我们在dp表中表示不出来，那我们就在计算的时候单独拿出来这种情况计算，即i和j相邻的时候:
     * for(int i = s.length()-1; i>=0; i--){
     *     for(int j = i+1; j<s.length(); j++){
     *         if(s.charAt(i) == s.charAt(j)){
     *             //i和j相邻的时候
     *             if(j - i == 1){
     *                 dp[i][j] = true;
     *             }
     *             else{
     *                 dp[i][j] = dp[i+1][j-1]
     *             }
     *         }
     *         else{
     *             dp[i][j] = false;
     *         }
     *     }
     * }
     * 由于最终需要输出最长的回文子串的数量，我们在遍历的时候记录一下即可。
     * *****************************************************************************************************************
     * 我觉得有必要解释一下为什么从右下角遍历：因为在填dp表时，(i, j) 位置的值依赖于（i+1,j-1），也就是当前位置的左下方。
     * 显然如果从上往下遍历，左下方的值就完全没有初始化，当然当前位置也会是错误的。但是从右下角遍历就保证了左下方的所有值都已经计算好了。
     * *****************************************************************************************************************
     * “回文串”是一个正读和反读都一样的字符串，比如“level”或者“noon”等等就是回文串。
     */
    public int countSubstrings(String s) {
        if (s == null || s.equals("")) {
            return 0;
        }
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        int result = s.length();
        for (int i = 0; i < n; i++) dp[i][i] = true;
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j - i == 1) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                } else {
                    dp[i][j] = false;
                }
                if (dp[i][j]) {
                    result++;
                }
            }
        }
        return result;

    }

    /**
     * 直接int数组，默认都是0，只需要置1确定是不是回文了，，，省去了赋值false哈哈哈
     * boolen数组也是默认false的 其实加上这个赋值只是为了可读性
     * *****************************************************************************************************************
     * “回文串”是一个正读和反读都一样的字符串，比如“level”或者“noon”等等就是回文串。
     * 百度百科 https://baike.baidu.com/item/%E5%9B%9E%E6%96%87%E4%B8%B2/1274921?fr=aladdin
     */
    public int countSubstrings2(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        int nums = s.length();

        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j - i < 3) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if (dp[i][j] == 1) {
                    nums++;
                }
            }
        }
        return nums;
    }

    /**
     * https://baike.baidu.com/item/%E5%9B%9E%E6%96%87%E4%B8%B2/1274921?fr=aladdin
     * *****************************************************************************************************************
     * 算法编辑
     * 1、初始化标志flag=true；
     * 2、输入字符串str，并获取其长度len；
     * 3、定义并初始化游标i=0，j=len-1，分别指向字符串开头和末尾；
     * 4、比较字符str[i]和str[j],若i==j，转至7，否则往下执行5；
     * 5、若str[i]和str[j]相等，则游标i加1，游标j减1后转至4，否则往下执行6；
     * 6、令标志位flag=flase，结束比较，str不是回文串，算法结束。
     * 7、若str[i]和str[j]相等，结束比较，flag=true,str为回文串，算法结束。
     * *****************************************************************************************************************
     * Java实现该算法
     */
    public static boolean check(String str) {
        if (null == str || "".equals(str)) {
            return false;
        }
        int i = 0;
        int j = str.length() - 1;
        String[] strings = str.split("");
        boolean flag = false;
        for (; i <= j; i++, j--) {
            if (!strings[i].equals(strings[j])) {
                return false;
            }
        }
        return true;
    }

    /**
     * 647. Palindromic Substrings
     * Given a string, your task is to count how many palindromic substrings in this string.
     * The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.
     * *****************************************************************************************************************
     * Example 1:
     * Input: "abc"
     * Output: 3
     * Explanation: Three palindromic strings: "a", "b", "c".
     * Example 2:
     * Input: "aaa"
     * Output: 6
     * Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
     * *****************************************************************************************************************
     * https://leetcode.com/problems/palindromic-substrings/discuss/105689/Java-solution-8-lines-extendPalindrome
     * Idea is start from each index and try to extend palindrome for both odd and even length.
     * *****************************************************************************************************************
     * It is not recommended to have a class level variables, like count here.
     * You can return an int count in the extendPalindrome() and then do summation of both in the main loop and then return the final count of all palindromes.
     * *****************************************************************************************************************
     * The extendPalindrome method could be simplified as:
     * private void expand(String s, int i, int j) {
     *     while (i >= 0 && j < s.length() && s.charAt(i--) == s.charAt(j++)) count++;
     * }
     * *****************************************************************************************************************
     * 图解如下:
     * https://leetcode.com/problems/palindromic-substrings/discuss/105688/Very-Simple-Java-Solution-with-Detail-Explanation
     */
    public int countSubstrings3(String s) {
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            count += helper(s, i, i);
            count += helper(s, i, i + 1);
        }
        return count;
    }

    public int helper(String s, int left, int right) {
        int count = 0;
        while ((left >= 0 && right <= s.length() - 1) && (s.charAt(left) == s.charAt(right))) {
            left--;
            right++;
            count++;
        }
        return count;
    }

    /**
     * https://leetcode-cn.com/problems/palindromic-substrings/solution/shui-kan-shui-xi-huan-by-lora-h-wr3k/
     * [Java] Simple Code: DP, short
     * https://leetcode.com/problems/palindromic-substrings/discuss/258917/Java-Simple-Code%3A-DP-short
     * 	  0  1  2  3  4
     * i\j | a  b  b  a  c
     * --------------------
     * 0 a | T  F  F  T  F
     * 1 b | F  T  T  F  F
     * 2 b | F  F  T  F  F
     * 3 a | F  F  F  T  F
     * 4 c | F  F  F  F  T
     * In the above table, P(i,j) is to represent if a substring s within the range [i,j] is a palindrome or not.
     *
     * If i>j then P(i,j) = false. That's why we see the part below the diagonal is filled by F values. We don't have to compute this part.
     *
     * if i==j then P(i,j) = true. That's why we see every point on the main diagonal is T
     *
     * if i<j then we will compare 2 heads at i and j:
     *
     * if s.charAt(i) != s.charAt(j) then P(i,j) = false
     * if s.charAt(i) == s.charAt(j) then P(i,j) = P(i+1,j-1).
     * So, if somehow P(i+1, j-1) is computed before P(i,j), we can solve this problem by DP solution.
     *
     * In order to do that we must iterate the part above the main diagonal by traversing all sub diagonals.
     *
     * main diagonal: T T T T T
     * sub diagonal:  F T F F
     * sub diagonal:  F F F
     * sub diagonal:  T F
     * sub diagonal:  F
     * We come up with the following code:
     * *****************************************************************************************************************
     * Can you pls explain the condition :
     * dp[i][j] = (i+1 >= j-1) ? true : dp[i+1][j-1]
     * *****************************************************************************************************************
     * As I explained, dp[i][j] is to represent if a substring s within the range [i,j] is a palindrome or not. If i+1 >= j-1, there are 2 cases:
     *
     * i+1 == j-1 => j-i == 2, e.g: s.substring(i,j) = "aba", s.charAt(i) == s.charAt(j) == 'a', s.charAt(i+1) == s.charAt(j-1) == 'b', it's always a palindrome
     * i+1 > j-1 => 2 > j-i > 0 => j-i == 1, e.g: s.susbstring(i,j) = "aa", it's also always a palindrome.
     * We come up to if (i+1 >= j-1) then dp[i][j] = true
     */
    public int countSubstrings5(String s) {
        int count = 0, n = s.length();
        boolean[][] dp = new boolean[n][n];
        for (int d = 0; d < n; d++) {
            for (int i = 0; i + d < n; i++) {
                int j = i + d;
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = (i + 1 >= j - 1) ? true : dp[i + 1][j - 1];
                    if (dp[i][j]) count++;
                }
            }
        }
        return count;
    }


    /**
     * Same idea in C++.
     * public:
     *     int countSubstrings(string s) {
     *         int counter = 0, n = s.size();
     *         for (int i = 0; i < n; i++) {
     *             for (int l = i, r = i; l >= 0 && r < n && s[l] == s[r]; l--, r++) {
     *                 counter++;
     *             }
     *             for (int l = i, r = i + 1; l >= 0 && r < n && s[l] == s[r]; l--, r++) {
     *                 counter++;
     *             }
     *         }
     *         return counter;
     *     }
     */
    public int countSubstrings4(String s) {
        int counter = 0, n = s.length();
        for (int i = 0; i < n; i++) {
            for (int l = i, r = i; l >= 0 && r < n && s.charAt(l) == s.charAt(r); l--, r++) {
                counter++;
            }
            for (int l = i, r = i + 1; l >= 0 && r < n && s.charAt(l) == s.charAt(r); l--, r++) {
                counter++;
            }
        }
        return counter;
    }

    /**
     * 5. 最长回文子串
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     * *****************************************************************************************************************
     * Example 1:
     * Input: s = "babad"
     * Output: "bab"
     * Note: "aba" is also a valid answer.
     *
     * Example 2:
     * Input: s = "cbbd"
     * Output: "bb"
     *
     * Example 3:
     * Input: s = "a"
     * Output: "a"
     *
     * Example 4:
     * Input: s = "ac"
     * Output: "a"
     * *****************************************************************************************************************
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/longest-palindromic-substring/solution/zhong-xin-kuo-san-dong-tai-gui-hua-by-liweiwei1419/
     * 动态规划、中心扩散、Manacher 算法
     * 说明：
     * 以下解法中「暴力算法」是基础，「动态规划」必须掌握，「中心扩散」方法要会写；
     * 「Manacher 算法」仅用于扩宽视野，绝大多数的算法面试中，面试官都不会要求写这个方法（除非面试者是竞赛选手）。
     * *****************************************************************************************************************
     * 方法一：暴力匹配 （Brute Force）
     * 根据回文子串的定义，枚举所有长度大于等于 22 的子串，依次判断它们是否是回文；
     * 在具体实现时，可以只针对大于“当前得到的最长回文子串长度”的子串进行“回文验证”；
     * 在记录最长回文子串的时候，可以只记录“当前子串的起始位置”和“子串长度”，不必做截取。这一步我们放在后面的方法中实现。
     * 说明：暴力解法时间复杂度高，但是思路清晰、编写简单。由于编写正确性的可能性很大，可以使用暴力匹配算法检验我们编写的其它算法是否正确。优化的解法在很多时候，是基于“暴力解法”，以空间换时间得到的，因此思考清楚暴力解法，分析其缺点，很多时候能为我们打开思路。
     * 参考代码 1：Java 代码正常运行，C++ 代码超出内存限制，Python 代码超时。
     */
    public String longestPalindrome(String s) {
        int len = s.length();
        if (len < 2) {
            return s;
        }

        int maxLen = 1;
        int begin = 0;
        // s.charAt(i) 每次都会检查数组下标越界，因此先转换成字符数组
        char[] charArray = s.toCharArray();

        // 枚举所有长度大于 1 的子串 charArray[i..j]
        for (int i = 0; i < len - 1; i++) {
            for (int j = i + 1; j < len; j++) {
                if (j - i + 1 > maxLen && validPalindromic(charArray, i, j)) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }

    /**
     * 验证子串 s[left..right] 是否为回文串
     */
    private boolean validPalindromic(char[] charArray, int left, int right) {
        while (left < right) {
            if (charArray[left] != charArray[right]) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    /**
     * 方法二：动态规划
     * 下面是「动态规划』问题的思考路径，供大家参考。
     *
     * 特别说明：
     *
     * 以下「动态规划」的解释只帮助大家了解「动态规划」问题的基本思想；
     * 「动态规划」问题可以难到非常难，在学习的时候建议不要钻到特别难的问题中去；
     * 掌握经典的动态规划问题的解法，理解状态的定义的由来、会列出状态转移方程；
     * 然后再配合适当难度的问题的练习；
     * 有时间和感兴趣的话可以做一些不太常见的类型的问题，拓宽视野；
     * 「动态规划」讲得比较好的经典书籍是《算法导论》。
     * 提示：右键「在新便签页打开图片」可查看大图。
     * *****************************************************************************************************************
     * 1、思考状态（重点）
     * 状态的定义，先尝试「题目问什么，就把什么设置为状态」；
     * 然后思考「状态如何转移」，如果「状态转移方程」不容易得到，尝试修改定义，目的依然是为了方便得到「状态转移方程」。
     * 「状态转移方程」是原始问题的不同规模的子问题的联系。即大问题的最优解如何由小问题的最优解得到。
     * 2、思考状态转移方程（核心、难点）
     * 状态转移方程是非常重要的，是动态规划的核心，也是难点；
     * 常见的推导技巧是：分类讨论。即：对状态空间进行分类；
     * 归纳「状态转移方程」是一个很灵活的事情，通常是具体问题具体分析；
     * 除了掌握经典的动态规划问题以外，还需要多做题；
     * 如果是针对面试，请自行把握难度。掌握常见问题的动态规划解法，理解动态规划解决问题，是从一个小规模问题出发，逐步得到大问题的解，并记录中间过程；
     * 「动态规划」方法依然是「空间换时间」思想的体现，常见的解决问题的过程很像在「填表」。
     * 3、思考初始化
     * 初始化是非常重要的，一步错，步步错。初始化状态一定要设置对，才可能得到正确的结果。
     * 角度 1：直接从状态的语义出发；
     * 角度 2：如果状态的语义不好思考，就考虑「状态转移方程」的边界需要什么样初始化的条件；
     * 角度 3：从「状态转移方程」方程的下标看是否需要多设置一行、一列表示「哨兵」（sentinel），这样可以避免一些特殊情况的讨论。
     * 4、思考输出
     * 有些时候是最后一个状态，有些时候可能会综合之前所有计算过的状态。
     * 5、思考优化空间（也可以叫做表格复用）
     * 「优化空间」会使得代码难于理解，且是的「状态」丢失原来的语义，初学的时候可以不一步到位。先把代码写正确是更重要；
     * 「优化空间」在有一种情况下是很有必要的，那就是状态空间非常庞大的时候（处理海量数据），此时空间不够用，就必须「优化空间」；
     * 非常经典的「优化空间」的典型问题是「0-1 背包」问题和「完全背包」问题。
     * （下面是这道问题「动态规划」方法的分析）
     * 这道题比较烦人的是判断回文子串。因此需要一种能够快速判断原字符串的所有子串是否是回文子串的方法，于是想到了「动态规划」。
     * 「动态规划」的一个关键的步骤是想清楚「状态如何转移」。事实上，「回文」天然具有「状态转移」性质。
     * 一个回文去掉两头以后，剩下的部分依然是回文（这里暂不讨论边界情况）；
     * 依然从回文串的定义展开讨论：
     * 如果一个字符串的头尾两个字符都不相等，那么这个字符串一定不是回文串；
     * 如果一个字符串的头尾两个字符相等，才有必要继续判断下去。
     * 如果里面的子串是回文，整体就是回文串；
     * 如果里面的子串不是回文串，整体就不是回文串。
     * 即：在头尾字符相等的情况下，里面子串的回文性质据定了整个子串的回文性质，这就是状态转移。因此可以把「状态」定义为原字符串的一个子串是否为回文子串。
     * 第 1 步：定义状态
     * dp[i][j] 表示子串 s[i..j] 是否为回文子串，这里子串 s[i..j] 定义为左闭右闭区间，可以取到 s[i] 和 s[j]。
     * 第 2 步：思考状态转移方程
     * 在这一步分类讨论（根据头尾字符是否相等），根据上面的分析得到：
     * dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]
     * 说明：
     * 「动态规划」事实上是在填一张二维表格，由于构成子串，因此 i 和 j 的关系是 i <= j ，因此，只需要填这张表格对角线以上的部分。
     * 看到 dp[i + 1][j - 1] 就得考虑边界情况。
     * 边界条件是：表达式 [i + 1, j - 1] 不构成区间，即长度严格小于 2，即 j - 1 - (i + 1) + 1 < 2 ，整理得 j - i < 3。
     * 这个结论很显然：j - i < 3 等价于 j - i + 1 < 4，即当子串 s[i..j] 的长度等于 2 或者等于 3 的时候，其实只需要判断一下头尾两个字符是否相等就可以直接下结论了。
     * 如果子串 s[i + 1..j - 1] 只有 1 个字符，即去掉两头，剩下中间部分只有 11 个字符，显然是回文；
     * 如果子串 s[i + 1..j - 1] 为空串，那么子串 s[i, j] 一定是回文子串。
     * 因此，在 s[i] == s[j] 成立和 j - i < 3 的前提下，直接可以下结论，dp[i][j] = true，否则才执行状态转移。
     * 第 3 步：考虑初始化
     * 初始化的时候，单个字符一定是回文串，因此把对角线先初始化为 true，即 dp[i][i] = true 。
     * 事实上，初始化的部分都可以省去。因为只有一个字符的时候一定是回文，dp[i][i] 根本不会被其它状态值所参考。
     * 第 4 步：考虑输出
     * 只要一得到 dp[i][j] = true，就记录子串的长度和起始位置，没有必要截取，这是因为截取字符串也要消耗性能，记录此时的回文子串的「起始位置」和「回文长度」即可。
     * 第 5 步：考虑优化空间
     * 因为在填表的过程中，只参考了左下方的数值。事实上可以优化，但是增加了代码编写和理解的难度，丢失可读和可解释性。在这里不优化空间。
     * 注意事项：总是先得到小子串的回文判定，然后大子串才能参考小子串的判断结果，即填表顺序很重要。
     * 大家能够可以自己动手，画一下表格，相信会对「动态规划」作为一种「表格法」有一个更好的理解。
     * 参考代码 2：
     *
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/longest-palindromic-substring/solution/zhong-xin-kuo-san-dong-tai-gui-hua-by-liweiwei1419/
     */
    public String longestPalindrome2(String s) {
        // 特判
        int len = s.length();
        if (len < 2) {
            return s;
        }

        int maxLen = 1;
        int begin = 0;

        // dp[i][j] 表示 s[i, j] 是否是回文串
        boolean[][] dp = new boolean[len][len];
        char[] charArray = s.toCharArray();

        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }
        for (int j = 1; j < len; j++) {
            for (int i = 0; i < j; i++) {
                if (charArray[i] != charArray[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                // 只要 dp[i][j] == true 成立，就表示子串 s[i..j] 是回文，此时记录回文长度和起始位置
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }

    /**
     * https://leetcode.com/problems/longest-palindromic-substring/discuss/2921/Share-my-Java-solution-using-dynamic-programming
     *
     * dp(i, j) represents whether s(i ... j) can form a palindromic substring, dp(i, j) is true when s(i) equals to s(j) and s(i+1 ... j-1) is a palindromic substring.
     * When we found a palindrome, check if it's the longest one. Time complexity O(n^2).
     */
    public String longestPalindrome3(String s) {
        int n = s.length();
        String res = null;

        boolean[][] dp = new boolean[n][n];

        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                dp[i][j] = s.charAt(i) == s.charAt(j) && (j - i < 3 || dp[i + 1][j - 1]);

                if (dp[i][j] && (res == null || j - i + 1 > res.length())) {
                    res = s.substring(i, j + 1);
                }
            }
        }

        return res;
    }

    /**
     * Thanks @jeantimex for this beautiful solution and explanation. Here is the same code with useful comments which helped me understand
     */
    public String longestPalindrome4(String s) {
        int n = s.length();
        String res = null;
        int palindromeStartsAt = 0, maxLen = 0;

        boolean[][] dp = new boolean[n][n];
        // dp[i][j] indicates whether substring s starting at index i and ending at j is palindrome

        for (int i = n - 1; i >= 0; i--) { // keep increasing the possible palindrome string
            for (int j = i; j < n; j++) { // find the max palindrome within this window of (i,j)

                //check if substring between (i,j) is palindrome
                dp[i][j] = (s.charAt(i) == s.charAt(j)) // chars at i and j should match
                        &&
                        (j - i < 3  // if window is less than or equal to 3, just end chars should match
                                || dp[i + 1][j - 1]); // if window is > 3, substring (i+1, j-1) should be palindrome too

                //update max palindrome string
                if (dp[i][j] && (j - i + 1 > maxLen)) {
                    palindromeStartsAt = i;
                    maxLen = j - i + 1;
                }
            }
        }
        return s.substring(palindromeStartsAt, palindromeStartsAt + maxLen);
    }

    /**
     * Bottom-up DP Logical Thinking
     * Intuitively, we list all the substrings, pick those palindromic, and get the longest one. However, that causes TLE for we reach the same situations (input substrings) many times.
     *
     * To optimize, we decompose the problem as follows
     *
     * state variable:
     * start index and end index of a substring can identify a state, which influences our decision
     * so state variable is state(s, e) indicates whether str[s, e] is palindromic
     * goal state:
     * max(e - s + 1) that makes state(s, e) = true
     * state transition:
     * Let's observe example base cases
     *
     * for s = e, "a" is palindromic,
     * for s + 1 = e, "aa" is palindromic (if str[s] = str[e])
     * for s + 2 = e, "aba" is palindromic (if str[s] = str[e] and "b" is palindromic)
     * for s + 3 = e, "abba" is palindromic (if str[s] = str[e] and "bb" is palindromic)
     * we realize that
     *
     * for s + dist = e, str[s, e] will be palindromic (if str[s] == str[e] and str[s + 1, e - 1] is palindromic)
     * state transition equation:
     *
     * state(s, e) is true:
     * for s = e,
     * for s + 1 = e,  if str[s] == str[e]
     * for s + 2 <= e, if str[s] == str[e] && state(s + 1, e - 1) is true
     * note:
     * state(s + 1, e - 1) should be calculated before state(s, e). That is, s is decreasing during the bottop-up dp implementation, while the dist between s and e is increasing, that's why
     *
     *         for (int s = len - 1; s >= 0; s--) {
     *             for (int dist = 1; dist < len - i; dist++) {
     * We keep track of longestPalindromeStart, longestPalindromeLength for the final output.
     */
    public String longestPalindrome5(String s) {
        // Corner cases.
        if (s.length() <= 1) return s;

        int len = s.length(), longestPalindromeStart = 0, longestPalindromeLength = 1;
        // state[i][j] true if s[i, j] is palindrome.
        boolean[][] state = new boolean[len][len];

        // Base cases.
        for (int i = 0; i < len; i++) {
            state[i][i] = true; // dist = 0.
        }

        for (int i = len - 1; i >= 0; i--) {
            for (int dist = 1; dist < len - i; dist++) {
                int j = dist + i;
                state[i][j] = (dist == 1) ? s.charAt(i) == s.charAt(j) : (s.charAt(i) == s.charAt(j)) && state[i + 1][j - 1];
                if (state[i][j] && j - i + 1 > longestPalindromeLength) {
                    longestPalindromeLength = j - i + 1;
                    longestPalindromeStart = i;
                }
            }
        }

        return s.substring(longestPalindromeStart, longestPalindromeStart + longestPalindromeLength);
    }
    /**
     * If you don't like dist:
     */
    public String longestPalindrome6(String s) {
        if (s.length() <= 1)
            return s;

        boolean[][] dp = new boolean[s.length()][s.length()];

        for (int i = 0; i < s.length(); i++)
            dp[i][i] = true;

        int longestPalindromeStart = 0, longestPalindromeLength = 1;
        for (int start = s.length() - 1; start >= 0; start--) {
            for (int end = start + 1; end < s.length(); end++) {
                if (s.charAt(start) == s.charAt(end)) {
                    if (end - start == 1 || dp[start + 1][end - 1]) {
                        dp[start][end] = true;
                        if (longestPalindromeLength < end - start + 1) {
                            longestPalindromeStart = start;
                            longestPalindromeLength = end - start + 1;
                        }
                    }
                }

            }
        }

        return s.substring(longestPalindromeStart, longestPalindromeStart + longestPalindromeLength);
    }

    /**
     * 32. 最长有效括号（字节跳动、亚马逊、微软在半年内面试中考过）（困难）
     * 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
     * *****************************************************************************************************************
     * 示例 1：
     *
     * 输入：s = "(()"
     * 输出：2
     * 解释：最长有效括号子串是 "()"
     * 示例 2：
     *
     * 输入：s = ")()())"
     * 输出：4
     * 解释：最长有效括号子串是 "()()"
     * 示例 3：
     *
     * 输入：s = ""
     * 输出：0
     * *****************************************************************************************************************
     * 解题思路一：常规
     * 对于这种括号匹配问题，一般都是使用栈
     * 我们先找到所有可以匹配的索引号，然后找出最长连续数列！
     * 例如：s = )(()())，我们用栈可以找到，
     * 位置 2 和位置 3 匹配，
     * 位置 4 和位置 5 匹配，
     * 位置 1 和位置 6 匹配，
     * 这个数组为：2,3,4,5,1,6 这是通过栈找到的，我们按递增排序！1,2,3,4,5,6
     * 找出该数组的最长连续数列的长度就是最长有效括号长度！
     * 所以时间复杂度来自排序：O(nlogn)。
     * 接下来我们思考，是否可以省略排序的过程,在弹栈时候进行操作呢?
     * 直接看代码理解!所以时间复杂度为：O(n)。
     * 解题思路二：dp 方法
     * 我们用 dp[i] 表示以 i 结尾的最长有效括号；
     * 1.当 s[i] 为 (,dp[i] 必然等于 0，因为不可能组成有效的括号；
     * 2.那么 s[i] 为 )
     * 2.1 当 s[i-1] 为 (，那么 dp[i] = dp[i-2] + 2；
     * 2.2 当 s[i-1] 为 ) 并且 s[i-dp[i-1] - 1] 为 (，那么 dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2]；
     * 时间复杂度：O(n)。
     * *****************************************************************************************************************
     * https://leetcode-cn.com/problems/longest-valid-parentheses/solution/zui-chang-you-xiao-gua-hao-by-powcai/
     */
    public int longestValidParentheses(String s) {
        if (s == null || s.length() == 0) return 0;
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(-1);
        //System.out.println(stack);
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') stack.push(i);
            else {
                stack.pop();
                if (stack.isEmpty()) stack.push(i);
                else {
                    res = Math.max(res, i - stack.peek());
                }
            }
        }
        return res;
    }


    public int longestValidParentheses2(String s) {
        if (s == null || s.length() == 0) return 0;
        int[] dp = new int[s.length()];
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            if (i > 0 && s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i - 2 >= 0 ? dp[i - 2] + 2 : 2);
                } else if (s.charAt(i - 1) == ')' && i - dp[i - 1] - 1 >= 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = dp[i - 1] + 2 + (i - dp[i - 1] - 2 >= 0 ? dp[i - dp[i - 1] - 2] : 0);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    /**
     * 72. 编辑距离（字节跳动、亚马逊、谷歌在半年内面试中考过）（困难）
     * 给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数。
     * 你可以对一个单词进行如下三种操作：
     * 插入一个字符
     * 删除一个字符
     * 替换一个字符
     * *****************************************************************************************************************
     * 示例 1：
     * 输入：word1 = "horse", word2 = "ros"
     * 输出：3
     * 解释：
     * horse -> rorse (将 'h' 替换为 'r')
     * rorse -> rose (删除 'r')
     * rose -> ros (删除 'e')
     * 示例 2：
     * 输入：word1 = "intention", word2 = "execution"
     * 输出：5
     * 解释：
     * intention -> inention (删除 't')
     * inention -> enention (将 'i' 替换为 'e')
     * enention -> exention (将 'n' 替换为 'x')
     * exention -> exection (将 'n' 替换为 'c')
     * exection -> execution (插入 'u')
     *
     * 来源：力扣（LeetCode）
     * 链接：https://leetcode-cn.com/problems/edit-distance
     * 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     * *****************************************************************************************************************
     * 自底向上 和自顶向下
     * 动态规划：
     * dp[i][j] 代表 word1 到 i 位置转换成 word2 到 j 位置需要最少步数
     * 所以，
     * 当 word1[i] == word2[j]，dp[i][j] = dp[i-1][j-1]；
     * 当 word1[i] != word2[j]，dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
     * 其中，dp[i-1][j-1] 表示替换操作，dp[i-1][j] 表示删除操作，dp[i][j-1] 表示插入操作。
     * 注意，针对第一行，第一列要单独考虑，我们引入 '' 下图所示：
     * 链接：https://leetcode-cn.com/problems/edit-distance/solution/zi-di-xiang-shang-he-zi-ding-xiang-xia-by-powcai-3/
     * 第一行，是 word1 为空变成 word2 最少步数，就是插入操作
     * 第一列，是 word2 为空，需要的最少步数，就是删除操作
     * 再附上自顶向下的方法，大家可以提供 Java 版吗？
     *
     * 作者：powcai
     * 链接：https://leetcode-cn.com/problems/edit-distance/solution/zi-di-xiang-shang-he-zi-ding-xiang-xia-by-powcai-3/
     * 来源：力扣（LeetCode）
     * 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
     */
    public int minDistance(String word1, String word2) {
        return 0;
    }
    /**
     * *****************************************************************************************************************
     * 第6周 第12课 | 动态规划（二）
     * 1. 实战题目解析：三角形最小路径和
     * *****************************************************************************************************************
     * 120. 三角形最小路径和（亚马逊、苹果、字节跳动在半年内面试考过）
     * 给定一个三角形 triangle ，找出自顶向下的最小路径和。
     * 每一步只能移动到下一行中相邻的结点上。
     * 相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
     * 示例 1：
     * 输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
     * 输出：11
     * 解释：自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
     * *****************************************************************************************************************
     * DP
     * a.重复性(分治) problem(i,j) = min(sub(i+1,j) , sub(i+1,j+1)) + a[i,j]
     * b.定义状态数组 f[i,j]
     * c.DP方程     f[i,j] = min(f[i+1,j] , f[i+1,j+1]) + a[i,j]
     * *****************************************************************************************************************
     * https://leetcode-cn.com/problems/triangle/solution/di-gui-ji-yi-hua-dp-bi-xu-miao-dong-by-sweetiee/
     * 解法一：递归
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        return dfs(triangle, 0, 0);
    }

    private int dfs(List<List<Integer>> triangle, int i, int j) {
        if (i == triangle.size()) {
            return 0;
        }
        return Math.min(dfs(triangle, i + 1, j), dfs(triangle, i + 1, j + 1)) + triangle.get(i).get(j);
    }
    /**
     * 解法二：递归 + 记忆化
     * 暴力搜索会有大量的重复计算，导致 超时，因此在 解法二 中结合记忆化数组进行优化。
     */
    Integer[][] memo;
    public int minimumTotal2(List<List<Integer>> triangle) {
        memo = new Integer[triangle.size()][triangle.size()];
        return  dfs2(triangle, 0, 0);
    }

    private int dfs2(List<List<Integer>> triangle, int i, int j) {
        if (i == triangle.size()) {
            return 0;
        }
        if (memo[i][j] != null) {
            return memo[i][j];
        }
        return memo[i][j] = Math.min(dfs(triangle, i + 1, j), dfs2(triangle, i + 1, j + 1)) + triangle.get(i).get(j);
    }
    /**
     * 解法三：动态规划
     * 定义二维 dp 数组，将解法二中「自顶向下的递归」改为「自底向上的递推」。
     * 1、状态定义：
     * dp[i][j] 表示从点 (i, j) 到底边的最小路径和。
     * 2、状态转移：
     * dp[i][j] = min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle[i][j]
     * 3、代码实现：
     * 链接：https://leetcode-cn.com/problems/triangle/solution/di-gui-ji-yi-hua-dp-bi-xu-miao-dong-by-sweetiee/
     */
    public int minimumTotal3(List<List<Integer>> triangle) {
        int n = triangle.size();
        // dp[i][j] 表示从点 (i, j) 到底边的最小路径和。
        int[][] dp = new int[n + 1][n + 1];
        // 从三角形的最后一行开始递推。
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                dp[i][j] = Math.min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0][0];
    }

    /**
     * 超哥的方法4:优先用这种方法   ***----***
     */
    public int minimumTotal4(List<List<Integer>> triangle) {
        int[] A = new int[triangle.size() + 1];
        // 从三角形的最后一行开始递推。
        for (int i = triangle.size() - 1; i >= 0; i--) {
            for (int j = 0; j < triangle.get(i).size(); j++) {
                A[j] = Math.min(A[j], A[j + 1]) + triangle.get(i).get(j);
            }
        }
        return A[0];
    }


    /**
     * 超哥的方法5: 递归 + 分治
     * 递归，自顶向下[超时]
     */
    int row;
    public int minimumTotal5(List<List<Integer>> triangle) {
        row = triangle.size();
        return helper(0, 0, triangle);
    }

    private int helper(int level, int c, List<List<Integer>> triangle) {
        System.out.println("helper: level=" + level + " c=" + c);
        if (level == row - 1) {
            return triangle.get(level).get(c);
        }
        int left = helper(level + 1, c, triangle);
        int right = helper(level + 1, c + 1, triangle);
        return Math.min(left, right) + triangle.get(level).get(c);
    }

    /**
     * 超哥的方法6: 递归 + 分治
     * 自顶向下,记忆化搜索[AC]
     */
    public int minimumTotal6(List<List<Integer>> triangle) {
        row = triangle.size();
        memo = new Integer[row][row];
        return helper2(0, 0, triangle);
    }
    private int helper2(int level, int j, List<List<Integer>> triangle) {
        System.out.println("helper: level=" + level + " j=" + j);
        if (memo[level][j] != null) {
            return memo[level][j];
        }
        if (level == row - 1) {
            return memo[level][j] = triangle.get(level).get(j);
        }
        int left = helper2(level + 1, j, triangle);
        int right = helper2(level + 1, j + 1, triangle);
        return memo[level][j] = Math.min(left, right) + triangle.get(level).get(j);
    }

    /**
     * 超哥的方法7: 递归 + 分治
     * 自底向上 , DP [AC]
     */
    public int minimumTotal7(List<List<Integer>> triangle) {
        row = triangle.size();
        int[] minlen = new int[row + 1];
        for (int level = row - 1; level >= 0; level--) {
            for (int j = 0; j <= level; j++) { // 第i行有i+1个数字,所以j多一列即j+1;
                minlen[j] = Math.min(minlen[j], minlen[j + 1]) + triangle.get(level).get(j);//对比上面的dp[i][j] = Math.min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle.get(i).get(j) 方法
            }
        }
        return minlen[0];
    }

    /**
     * Python easy to understand solutions (top-down, bottom-up).
     * # O(n*n/2) space, top-down
     * def minimumTotal1(self, triangle):
     *     if not triangle:
     *         return
     *     res = [[0 for i in xrange(len(row))] for row in triangle]
     *     res[0][0] = triangle[0][0]
     *     for i in xrange(1, len(triangle)):
     *         for j in xrange(len(triangle[i])):
     *             if j == 0:
     *                 res[i][j] = res[i-1][j] + triangle[i][j]
     *             elif j == len(triangle[i])-1:
     *                 res[i][j] = res[i-1][j-1] + triangle[i][j]
     *             else:
     *                 res[i][j] = min(res[i-1][j-1], res[i-1][j]) + triangle[i][j]
     *     return min(res[-1])
     *
     * # Modify the original triangle, top-down
     * def minimumTotal2(self, triangle):
     *     if not triangle:
     *         return
     *     for i in xrange(1, len(triangle)):
     *         for j in xrange(len(triangle[i])):
     *             if j == 0:
     *                 triangle[i][j] += triangle[i-1][j]
     *             elif j == len(triangle[i])-1:
     *                 triangle[i][j] += triangle[i-1][j-1]
     *             else:
     *                 triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])
     *     return min(triangle[-1])
     *
     * # Modify the original triangle, bottom-up
     * def minimumTotal3(self, triangle):
     *     if not triangle:
     *         return
     *     for i in xrange(len(triangle)-2, -1, -1):
     *         for j in xrange(len(triangle[i])):
     *             triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
     *     return triangle[0][0]
     *
     * # bottom-up, O(n) space
     * def minimumTotal(self, triangle):
     *     if not triangle:
     *         return
     *     res = triangle[-1]
     *     for i in xrange(len(triangle)-2, -1, -1):
     *         for j in xrange(len(triangle[i])):
     *             res[j] = min(res[j], res[j+1]) + triangle[i][j]
     *     return res[0]
     */

    /**
     * 53. 最大子序和（亚马逊、字节跳动在半年内面试常考）
     * 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     * 输入: [-2,1,-3,4,-1,2,1,-5,4]
     * 输出: 6
     * 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6
     * *****************************************************************************************************************
     * DP
     * a.重复性(分治) problem(i,j) = min(sub(i+1,j) , sub(i+1,j+1)) + a[i,j]
     * b.定义状态数组 f[i,j]
     * c.DP方程     f[i,j] = min(f[i+1,j] , f[i+1,j+1]) + a[i,j]
     * *****************************************************************************************************************
     * DP
     * a.重复性(分治) max_sum(i) = Max(max_sum(i-1) , 0) + a[i]
     * b.定义状态数组 f[i]
     * c.DP方程     f[i] = Max(f[i-1] , 0) + a[i]
     * *****************************************************************************************************************
     * dp[i] = nums[i] + max(0,dp[i-1])
     * return max(dp);
     * nums[i] = nums[i] + max(0,nums[i-1])
     * return max(nums);
     */
    public int maxSubArray(int[] nums) {
        int pre = 0, maxAns = nums[0];
        for (int x : nums) {
            pre = Math.max(pre + x, x);
            maxAns = Math.max(maxAns, pre);
        }
        return maxAns;
    }

    /**
     * 53. Maximum Subarray
     * Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
     * Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.
     * *****************************************************************************************************************
     * Analysis of this problem:
     * Apparently, this is a optimization problem, which can be usually solved by DP. So when it comes to DP, the first thing for us to figure out is the format of the sub problem(or the state of each sub problem). The format of the sub problem can be helpful when we are trying to come up with the recursive relation.
     * At first, I think the sub problem should look like: maxSubArray(int A[], int i, int j), which means the maxSubArray for A[i: j]. In this way, our goal is to figure out what maxSubArray(A, 0, A.length - 1) is. However, if we define the format of the sub problem in this way, it's hard to find the connection from the sub problem to the original problem(at least for me). In other words, I can't find a way to divided the original problem into the sub problems and use the solutions of the sub problems to somehow create the solution of the original one.
     * So I change the format of the sub problem into something like: maxSubArray(int A[], int i), which means the maxSubArray for A[0:i ] which must has A[i] as the end element. Note that now the sub problem's format is less flexible and less powerful than the previous one because there's a limitation that A[i] should be contained in that sequence and we have to keep track of each solution of the sub problem to update the global optimal value. However, now the connect between the sub problem & the original one becomes clearer:
     * maxSubArray(A, i) = maxSubArray(A, i - 1) > 0 ? maxSubArray(A, i - 1) : 0 + A[i];
     * And here's the code
     * *****************************************************************************************************************
     * https://leetcode.com/problems/maximum-subarray/discuss/20193/DP-solution-and-some-thoughts
     */
    public int maxSubArray2(int[] A) {
        int n = A.length;
        int[] dp = new int[n];//dp[i] means the maximum subarray ending with A[i];
        dp[0] = A[0];
        int max = dp[0];
        for (int i = 1; i < n; i++) {
            dp[i] = A[i] + (Math.max(dp[i - 1], 0));//dp[i] = A[i] + (dp[i - 1] > 0 ? dp[i - 1] : 0);
            max = Math.max(max, dp[i]);
        }

        return max;
    }

    /**
     * Accepted O(n) solution in java
     * this problem was discussed by Jon Bentley (Sep. 1984 Vol. 27 No. 9 Communications of the ACM P885)
     * the paragraph below was copied from his paper (with a little modifications)
     * algorithm that operates on arrays: it starts at the left end (element A[1]) and scans through to the right end (element A[n]), keeping track of the maximum sum subvector seen so far. The maximum is initially A[0]. Suppose we've solved the problem for A[1 .. i - 1]; how can we extend that to A[1 .. i]? The maximum
     * sum in the first I elements is either the maximum sum in the first i - 1 elements (which we'll call MaxSoFar), or it is that of a subvector that ends in position i (which we'll call MaxEndingHere).
     * MaxEndingHere is either A[i] plus the previous MaxEndingHere, or just A[i], whichever is larger.
     * *****************************************************************************************************************
     * https://leetcode.com/problems/maximum-subarray/discuss/20211/Accepted-O(n)-solution-in-java
     * *****************************************************************************************************************
     * interesting fact: a problem solved by a computer professor in 1984, now become an easy level problem in Leetcode in 2016/2017...
     */
    public int maxSubArray3(int[] A) {
        int maxSoFar = A[0], maxEndingHere = A[0];
        for (int i = 1; i < A.length; ++i) {
            maxEndingHere = Math.max(maxEndingHere + A[i], A[i]);
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }
        return maxSoFar;
    }

    /**
     * 152. 乘积最大子数组（亚马逊、字节跳动、谷歌在半年内面试中考过）
     * 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
     * 输入: [2,3,-2,4]
     * 输出: 6
     * 解释: 子数组 [2,3] 有最大乘积 6。
     * *****************************************************************************************************************
     * 思路
     * 这个问题很像「力扣」第 53 题：最大子序和，只不过当前这个问题求的是乘积的最大值；
     * 「连续」这个概念很重要，可以参考第 53 题的状态设计，将状态设计为：以 nums[i]结尾的连续子数组的最大值；
     * 类似状态设计的问题还有「力扣」第 300 题：最长上升子序列，「子数组」、「子序列」问题的状态设计的特点是：以 nums[i] 结尾，这是一个经验，可以简化讨论。
     * 提示：以 nums[i] 结尾这件事情很重要，贯穿整个解题过程始终，请大家留意。
     * 分析与第 53 题的差异
     * 求乘积的最大值，示例中负数的出现，告诉我们这题和 53 题不一样了，一个正数乘以负数就变成负数，即：最大值乘以负数就变成了最小值；
     * 因此：最大值和最小值是相互转换的，这一点提示我们可以把这种转换关系设计到「状态转移方程」里去；
     * 如何解决这个问题呢？这里常见的技巧是在「状态设计」的时候，在原始的状态设计后面多加一个维度，减少分类讨论，降低解决问题的难度。
     * 这里是百度百科的「无后效性」词条的解释：
     * 无后效性是指如果在某个阶段上过程的状态已知，则从此阶段以后过程的发展变化仅与此阶段的状态有关，而与过程在此阶段以前的阶段所经历过的状态无关。利用动态规划方法求解多阶段决策过程问题，过程的状态必须具备无后效性。
     * 再翻译一下就是：「动态规划」通常不关心过程，只关心「阶段结果」，这个「阶段结果」就是我们设计的「状态」。什么算法关心过程呢？「回溯算法」，「回溯算法」需要记录过程，复杂度通常较高。
     * 而将状态定义得更具体，通常来说对于一个问题的解决是满足「无后效性」的。这一点的叙述很理论化，不熟悉朋友可以通过多做相关的问题来理解「无后效性」这个概念。
     * *****************************************************************************************************************
     * 下面提供一些问题：
     * 滑动窗口典型问题：第 3 题、第 76 题、第 438 题；
     * 双指针典型问题：第 11 题、第 15 题、第 42 题。
     * 你可以在做练习的过程中体会一下，「滑动窗口」和「双指针」在本质上「暴力解法」的优化，使用两个变量同向移动或者相向移动的过程中，排除了很多不必要的区间，但依然不丢失最优解。这道题没有这样的特点。
     * 一般问最优解而不问最优解怎么来的，先往 dp 上靠，然后尝试一下什么广度优先遍历啊，贪心啥的，这个话题很大啊，很多时候要靠一点感觉和经验。
     * 个人建议，仅供参考。
     * *****************************************************************************************************************
     * 第 1 步：状态设计（特别重要）
     * dp[i][j]：以 nums[i] 结尾的连续子数组的最值，计算最大值还是最小值由 j 来表示，j 就两个值；
     * 当 j = 0 的时候，表示计算的是最小值；
     * 当 j = 1 的时候，表示计算的是最大值。
     * 这样一来，状态转移方程就容易写出。
     *
     * 第 2 步：推导状态转移方程（特别重要）
     * 由于状态的设计 nums[i] 必须被选取（请大家体会这一点，这一点恰恰好也是使得子数组、子序列问题更加简单的原因：当情况复杂、分类讨论比较多的时候，需要固定一些量，以简化计算）；
     * nums[i] 的正负和之前的状态值（正负）就产生了联系，由此关系写出状态转移方程：
     * ①当 nums[i] > 0 时，由于是乘积关系：
     * 最大值乘以正数依然是最大值；
     * 最小值乘以同一个正数依然是最小值；
     *
     * ②当 nums[i] < 0 时，依然是由于乘积关系：
     * 最大值乘以负数变成了最小值；
     * 最小值乘以同一个负数变成最大值；
     *
     * ③当 nums[i] = 0 的时候，由于 nums[i] 必须被选取，最大值和最小值都变成 0 ，合并到上面任意一种情况均成立。
     *
     * 但是，还要注意一点，之前状态值的正负也要考虑：例如，在考虑最大值的时候，当 nums[i] > 0 是，如果 dp[i - 1][1] < 0 （之前的状态最大值） ，此时 nums[i] 可以另起炉灶（这里依然是第 53 题的思想），此时 dp[i][1] = nums[i] ，合起来写就是：
     * dp[i][1] = max(nums[i], nums[i] * dp[i - 1][1]) if nums[i] >= 0
     * 其它三种情况可以类似写出，状态转移方程如下：
     * dp[i][0] = min(nums[i], nums[i] * dp[i - 1][0]) if nums[i] >= 0
     * dp[i][1] = max(nums[i], nums[i] * dp[i - 1][1]) if nums[i] >= 0
     *
     * dp[i][0] = min(nums[i], nums[i] * dp[i - 1][1]) if nums[i] < 0
     * dp[i][1] = max(nums[i], nums[i] * dp[i - 1][0]) if nums[i] < 0
     *
     * 第 3 步：考虑初始化
     * 由于 nums[i] 必须被选取，那么 dp[i][0] = nums[0]，dp[i][1] = nums[0]。
     *
     * 第 4 步：考虑输出
     * 题目问连续子数组的乘积最大值，这些值需要遍历 dp[i][1] 获得。
     * 参考代码 1：
     * 复杂度分析：
     * 时间复杂度：O(N)，这里 N 是数组的长度，遍历 2 次数组；
     * 空间复杂度：O(N)，状态数组的长度为 2N。
     * *****************************************************************************************************************
     * 链接：https://leetcode-cn.com/problems/maximum-product-subarray/solution/dong-tai-gui-hua-li-jie-wu-hou-xiao-xing-by-liweiw/
     *
     */
    public int maxProduct(int[] nums) {
        int len = nums.length;
        if (len == 0) {
            return 0;
        }

        // dp[i][0]：以 nums[i] 结尾的连续子数组的最小值
        // dp[i][1]：以 nums[i] 结尾的连续子数组的最大值
        int[][] dp = new int[len][2];
        dp[0][0] = nums[0];
        dp[0][1] = nums[0];
        for (int i = 1; i < len; i++) {
            if (nums[i] >= 0) {
                dp[i][0] = Math.min(nums[i], nums[i] * dp[i - 1][0]);
                dp[i][1] = Math.max(nums[i], nums[i] * dp[i - 1][1]);
            } else {
                dp[i][0] = Math.min(nums[i], nums[i] * dp[i - 1][1]);
                dp[i][1] = Math.max(nums[i], nums[i] * dp[i - 1][0]);
            }
        }

        // 只关心最大值，需要遍历
        int res = dp[0][1];
        for (int i = 1; i < len; i++) {
            res = Math.max(res, dp[i][1]);
        }
        return res;
    }

    /**
     * 问题做到这个地方，其实就可以了。下面介绍一些非必要但进阶的知识。
     *
     * 第 5 步：考虑表格复用
     *
     * 动态规划问题，基于「自底向上」、「空间换时间」的思想，通常是「填表格」，本题也不例外；
     * 由于通常只关心最后一个状态值，或者在状态转移的时候，当前值只参考了上一行的值，因此在填表的过程中，表格可以复用，常用的技巧有：
     * 1、滚动数组（当前行只参考了上一行的时候，可以只用 2 行表格完成全部的计算）；
     * 2、滚动变量（斐波拉契数列问题）。
     * 掌握非常重要的「表格复用」技巧，来自「0-1 背包问题」（弄清楚为什么要倒序填表）和「完全背包问题」（弄清楚为什么可以正向填表）；
     * 「表格复用」的合理性，只由「状态转移方程」决定，即当前状态值只参考了哪些部分的值。
     * 参考代码 2：
     * 复杂度分析：
     * 时间复杂度：O(N)，这里 N 是数组的长度，最值也在一次遍历的过程中计算了出来；
     * 空间复杂度：O(1)，只使用了常数变量。
     * *****************************************************************************************************************
     * 这里说一点题外话：除了基础的「0-1」背包问题和「完全背包」问题，需要掌握「表格复用」的技巧以外。在绝大多数情况下，在「力扣」上做的「动态规划」问题都可以不考虑「表格复用」。
     * 做题通常可以不先考虑优化空间（个人观点，仅供参考），理由如下：
     * 空间通常来说是用户不敏感的，并且在绝大多数情况下，空间成本低，我们写程序通常需要优先考虑时间复杂度最优；
     * 时间复杂度和空间复杂度通常来说不可能同时最优，所以我们经常看到的是优化解法思路都是「空间换时间」，这一点几乎贯穿了基础算法领域的绝大多数的算法设计思想；
     * 限制空间的思路，通常来说比较难，一般是在优化的过程中才考虑优化空间，在一些限制答题时间的场景下（例如面试），先写出一版正确的代码是更重要的，并且不优化空间的代码一般来说，可读性和可解释性更强。
     * 以上个人建议，仅供参考。
     * *****************************************************************************************************************
     * 总结
     * 动态规划问题通常用于计算多阶段决策问题的最优解。
     * 1.多阶段，是指解决一个问题有多个步骤；
     * 2.最优解，是指「最优子结构」。
     * 动态规划有三个概念很重要：
     * 1.重复子问题：因为重复计算，所以需要「空间换时间」，记录子问题的最优解；
     * 2.最优子结构：规模较大的问题的最优解，由各个子问题的最优解得到；
     * 3.无后效性（上面已经解释）。
     * *****************************************************************************************************************
     * 动态规划有两个特别关键的步骤：
     * 1.设计状态：
     * 1.1有些题目问啥，就设计成什么；
     * 1.2如果不行，只要有利于状态转移，很多时候，就可以设计成状态；
     * 1.3根据过往经验；
     * 1.4还有一部分问题是需要在思考的过程中调整的，例如本题。
     * 2.推导状态转移方程：通常是由问题本身决定的。
     * *****************************************************************************************************************
     * 动态规划问题思考的两个方向：
     * 1.自顶向下：即「递归 + 记忆化」，入门的时候优先考虑这样做；
     * 2.自底向上：即「递推」，从一个最小的问题开始，逐步得到最终规模问题的解。后面问题见得多了，优先考虑这样做，绝大部分动态规划问题可以「自底向上」通过递推得到。
     * *****************************************************************************************************************
     * 相关练习
     *「力扣」第 376 题：摆动序列（中等）；
     * 股票系列 6 道问题：区别仅在于题目加了不同的约束。一般来说有一个约束，就在「状态设计」的时候在后面多加一维，消除后效性，这个系列里最难的问题，也只有 2 个约束，因此状态设计最多 3 维。增加维度使得状态设计满足无后效性，是常见的解决问题的技巧。
     * 「力扣」第 121 题：买卖股票的最佳时机（简单）；
     * 「力扣」第 122 题：买卖股票的最佳时机 II（简单） ；
     * 「力扣」第 123 题：买卖股票的最佳时机 III（困难）；
     * 「力扣」第 188 题：买卖股票的最佳时机 IV（困难）；
     * 「力扣」第 309 题：最佳买卖股票时机含冷冻期（中等）；
     * 「力扣」第 714 题：买卖股票的最佳时机含手续费（中等）。
     * 打家劫舍系列的两道问题都很典型：
     * 「力扣」第 198 题：打家劫舍（简单），第 213 题：打家劫舍 II（中等） 基于这个问题分治（分类讨论）做；
     * 「力扣」第 337 题：打家劫舍 III（中等），树形 dp 的入门问题，依然是加一个维度，使得求解过程具有无后效性，使用后序遍历，完成计算。
     *
     * 链接：https://leetcode-cn.com/problems/maximum-product-subarray/solution/dong-tai-gui-hua-li-jie-wu-hou-xiao-xing-by-liweiw/
     */
    public int maxProduct2(int[] nums) {
        int len = nums.length;
        if (len == 0) {
            return 0;
        }

        int preMax = nums[0];
        int preMin = nums[0];

        // 滚动变量
        int curMax;
        int curMin;

        int res = nums[0];
        for (int i = 1; i < len; i++) {
            if (nums[i] >= 0) {
                curMax = Math.max(preMax * nums[i], nums[i]);
                curMin = Math.min(preMin * nums[i], nums[i]);
            } else {
                curMax = Math.max(preMin * nums[i], nums[i]);
                curMin = Math.min(preMax * nums[i], nums[i]);
            }
            res = Math.max(res, curMax);

            // 赋值滚动变量
            preMax = curMax;
            preMin = curMin;
        }
        return res;
    }


    /**
     * 152. Maximum Product Subarray
     * Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.
     * *****************************************************************************************************************
     * Example 1:
     * Input: [2,3,-2,4]
     * Output: 6
     * Explanation: [2,3] has the largest product 6.
     * Example 2:
     * Input: [-2,0,-1]
     * Output: 0
     * Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
     * https://leetcode.com/problems/maximum-product-subarray/
     * *****************************************************************************************************************
     * Possibly simplest solution with O(n) time complexity
     */
    public int maxProduct(int A[], int n) {
        // store the result that is the max we have found so far
        int r = A[0];

        // imax/imin stores the max/min product of
        // subarray that ends with the current number A[i]
        for (int i = 1, imax = r, imin = r; i < n; i++) {
            // multiplied by a negative makes big number smaller, small number bigger
            // so we redefine the extremums by swapping them
            if (A[i] < 0)
                swap(imax, imin);

            // max/min product for the current number is either the current number itself
            // or the max/min by the previous number times the current one
            imax = Math.max(A[i], imax * A[i]);
            imin = Math.min(A[i], imin * A[i]);

            // the newly computed max value is a candidate for our global result
            r = Math.max(r, imax);
        }
        return r;
    }

    private void swap(int imax, int imin) {
        System.out.println("互换前imax=" + imax + "   imin=" + imin);
        int temp;
        temp = imax;
        imax = imin;
        imin = temp;
        System.out.println("互换后imax=" + imax + "   imin=" + imin);
    }

    /**
     * Loop through the array, each time remember the max and min value for the previous product,
     * the most important thing is to update the max and min value: we have to compare among max * A[i], min * A[i] as well as A[i],
     * since this is product, a negative * negative could be positive.
     */
    public int maxProduct3(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }
        int max = A[0], min = A[0], result = A[0];
        for (int i = 1; i < A.length; i++) {
            int temp = max;
            max = Math.max(Math.max(max * A[i], min * A[i]), A[i]);
            min = Math.min(Math.min(temp * A[i], min * A[i]), A[i]);
            if (max > result) {
                result = max;
            }
        }
        return result;
    }

    /**
     * 322. 零钱兑换（亚马逊在半年内面试中常考）
     * 给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
     * 你可以认为每种硬币的数量是无限的。
     * *****************************************************************************************************************
     * 动态规划、完全背包、BFS（包含完全背包问题公式推导）
     * 方法一：动态规划与记忆化递归
     * 思路：分析最优子结构。根据示例 1：
     * 输入: coins = [1, 2, 5], amount = 11
     * 凑成面值为 11 的最少硬币个数可以由以下三者的最小值得到：
     *
     * 凑成面值为 10 的最少硬币个数 + 面值为 1 的这一枚硬币；
     * 凑成面值为 9 的最少硬币个数 + 面值为 2 的这一枚硬币；
     * 凑成面值为 6 的最少硬币个数 + 面值为 5 的这一枚硬币。
     * 即 dp[11] = min (dp[10] + 1, dp[9] + 1, dp[6] + 1)。
     *
     * 可以直接把问题的问法设计成状态。
     *
     * 第 1 步：定义「状态」。dp[i] ：凑齐总价值 i 需要的最少硬币个数；
     * 第 2 步：写出「状态转移方程」。根据对示例 1 的分析：
     *
     * dp[amount] = min(dp[amount], 1 + dp[amount - coins[i]]) for i in [0, len - 1] if coins[i] <= amount
     * 说明：感谢 @paau 朋友纠正状态转移方程。
     *
     * 注意：
     * 单枚硬币的面值首先要小于等于 当前要凑出来的面值；
     * 剩余的那个面值也要能够凑出来，例如：求 dp[11] 需要参考 dp[10]。如果不能凑出 dp[10]，则 dp[10] 应该等于一个不可能的值，可以设计为 11 + 1，也可以设计为 -1 ，它们的区别只是在编码的细节上不一样。
     * 再次强调：新状态的值要参考的值以前计算出来的「有效」状态值。因此，不妨先假设凑不出来，因为求的是小，所以设置一个不可能的数。
     * *****************************************************************************************************************
     * 参考代码 1：
     * 注意：要求的是「恰好凑出面值」，所以初始化的时候需要赋值为一个不可能的值：amount + 1。只有在有「正常值」的时候，「状态转移」才可以正常发生。
     * *****************************************************************************************************************
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/coin-change/solution/dong-tai-gui-hua-shi-yong-wan-quan-bei-bao-wen-ti-/
     */
    public int coinChange(int[] coins, int amount) {
        // 给 0 占位
        int[] dp = new int[amount + 1];

        // 注意：因为要比较的是最小值，这个不可能的值就得赋值成为一个最大值
        Arrays.fill(dp, amount + 1);

        // 理解 dp[0] = 0 的合理性，单独一枚硬币如果能够凑出面值，符合最优子结构
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (i - coin >= 0 && dp[i - coin] != amount + 1) {
                    dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
                }
            }
        }

        if (dp[amount] == amount + 1) {
            dp[amount] = -1;
        }
        return dp[amount];
    }

    /**
     * *****************************************************************************************************************
     * 「动态规划」是「自底向上」求解。事实上，可以 直接面对问题求解 ，即「自顶向下」，但是这样的问题有 重复子问题，需要缓存已经求解过的答案，这叫 记忆化递归。
     * 参考代码 2：
     * 注意：由于 -1 是一个特殊的、有意义状态值（题目要求不能使用给出硬币面值凑出的时候，返回 -1），因此初值赋值为 -2，表示还未计算出结果。
     * *****************************************************************************************************************
     * 复杂度分析：
     * 时间复杂度：O(N×amount)，这里 N 是可选硬币的种类数，amount 是题目输入的面值；
     * 空间复杂度：O(amount)，状态数组的大小为 amount。
     * *****************************************************************************************************************
     */
    public int coinChange2(int[] coins, int amount) {
        int[] memo = new int[amount + 1];
        Arrays.fill(memo, -2);
        Arrays.sort(coins);
        return dfs(coins, amount, memo);
    }

    private int dfs(int[] coins, int amount, int[] memo) {
        int res = Integer.MAX_VALUE;
        if (amount == 0) {
            return 0;
        }

        if (memo[amount] != -2) {
            return memo[amount];
        }

        for (int coin : coins) {
            if (amount - coin < 0) {
                break;
            }

            int subRes = dfs(coins, amount - coin, memo);
            if (subRes == -1) {
                continue;
            }
            res = Math.min(res, subRes + 1);
        }
        return memo[amount] = (res == Integer.MAX_VALUE) ? -1 : res;
    }

    /**
     * 方法二：广度优先遍历
     * 具体在纸上画一下，就知道这其实是一个在「图」上的最短路径问题，「广度优先遍历」是求解这一类问题的算法。广度优先遍历借助「队列」实现。[点击链接查看 图]
     * 注意：
     * 由于是「图」，有回路，所以需要一个 visited 数组，记录哪一些结点已经访问过。
     * 在添加到队列的时候，就得将 visited 数组对应的值设置为 true，否则可能会出现同一个元素多次入队的情况。
     * 参考代码 3：
     * *****************************************************************************************************************
     * 复杂度分析：（待纠正）
     *
     * 时间复杂度：O(amount)，这里 amount 是题目输入的面值；
     * 空间复杂度：O(amount)，队列的大小为 amount。
     *
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/coin-change/solution/dong-tai-gui-hua-shi-yong-wan-quan-bei-bao-wen-ti-/
     */
    public int coinChange3(int[] coins, int amount) {
        if (amount == 0) {
            return 0;
        }

        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[amount + 1];

        visited[amount] = true;
        queue.offer(amount);

        // 排序是为了加快广度优先遍历过程中，对硬币面值的遍历，起到剪枝的效果
        Arrays.sort(coins);

        int step = 1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Integer head = queue.poll();
                for (int coin : coins) {
                    int next = head - coin;
                    // 只要遇到 0，就找到了一个最短路径
                    if (next == 0) {
                        return step;
                    }

                    if (next < 0) {
                        // 由于 coins 升序排序，后面的面值会越来越大，剪枝
                        break;
                    }

                    if (!visited[next]) {
                        queue.offer(next);
                        // 添加到队列的时候，就应该立即设置为 true
                        // 否则还会发生重复访问
                        visited[next] = true;
                    }
                }
            }
            step++;
        }
        // 进入队列的顶点都出队，都没有看到 0 ，就表示凑不出当前面值
        return -1;
    }

    /**
     * 方法三：套「完全背包」问题的公式
     *
     * 为什么是「完全背包」问题：
     *
     * 每个硬币可以使用无限次；
     * 硬币总额有限制；
     * 并且具体组合是顺序无关的，还以示例 1 为例：面值总额为 11，方案 [1, 5, 5] 和方案 [5, 1, 5] 视为同一种方案。
     * 但是与「完全」背包问题不一样的地方是：
     *
     * 要求恰好填满容积为 amount 的背包，重点是「恰好」、「刚刚好」，而原始的「完全背包」问题只是要求「不超过」；
     * 题目问的是总的硬币数最少，原始的「完全背包」问题让我们求的是总价值最多。
     * 这一点可以认为是：每一个硬币有一个「占用空间」属性，并且值是固定的，固定值为 11；作为「占用空间」而言，考虑的最小化是有意义的。等价于把「完全背包」问题的「体积」和「价值」属性调换了一下。因此，这个问题的背景是「完全背包」问题。
     *
     * 可以使用「完全背包」问题的解题思路（「0-1 背包」问题也是这个思路）：
     *
     * 一个一个硬币去看，一点点扩大考虑的价值的范围（自底向上考虑问题的思想）。其实就是在不断地做尝试和比较，实际生活中，人也是这么干的，「盗贼」拿东西也是这样的，看到一个体积小，价值大的东西，就会从背包里把占用地方大，廉价的物品换出来。
     *
     * 所以在代码里：外层循环先遍历的是硬币面值，内层循环遍历的是面值总和。
     *
     * 说明：以下代码提供的是「完全背包」问题「最终版本」的代码。建议读者按照以下路径进行学习，相信就不难理解这个代码为什么这样写了。
     *
     * 「0-1 背包」问题，二维表格的写法；
     * 「0-1 背包」问题，滚动数组的写法；
     * 「0-1 背包」问题只用一行，从后向前覆盖赋值的写法（因为只关心最后一行最后一格数值，每个单元格只参考它上一行，并且是正上方以及正上方左边的单元格数值）；
     * 「完全背包」问题，二维表格的写法（最朴素的解法，枚举每个硬币可以选用的个数）；
     * 「完全背包」问题，优化了「状态转移方程」的二维表格的写法（每一行只参考了上一行正上方的数值，和当前行左边的数值）；
     * 「完全背包」问题压缩成一行的写法，正好与「0-1 背包」问题相反，「0-1 背包」问题倒着写，「完全背包」问题正着写（看看填表顺序，就明白了）。
     * （这里省略了 2 版代码，请读者自己学习背包问题的知识，将它们补上。）
     * 参考代码 4：
     *
     * 复杂度分析：（待纠正）
     * 时间复杂度：O(N×amount)，这里 N 是可选硬币的种类数，amount 是题目输入的面值；
     * 空间复杂度：O(amount)，状态数组的大小为 amount。
     * *****************************************************************************************************************
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/coin-change/solution/dong-tai-gui-hua-shi-yong-wan-quan-bei-bao-wen-ti-/
     */
    public int coinChange4(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;

        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }

        if (dp[amount] == amount + 1) {
            dp[amount] = -1;
        }
        return dp[amount];
    }

    /**
     * 附录：完全背包问题学习笔记
     *
     * 有 N 件物品和一个容量是 V 的背包。每件物品只能使用一次，每种物品都有无限件可用。第 i 件物品的体积是 wi ，价值是 vi 。求将哪些物品装入背包，可使这些物品的总体积不超过背包容量，且总价值最大。输出最大价值。
     *
     * 分析：「完全背包问题」的重点在于
     * 1.每种物品都有无限件可用；
     * 2.一个元素可以使用多个，且不计算顺序。
     * 状态定义（和 「0-1 背包问题一样」）：dp[i][j] 表示考虑物品区间 [0, i] 里，不超过背包容量，能够获得的最大价值；
     *
     * 状态转移方程：
     * dp[i][j]=max(dp[i−1][j],dp[i−1][j−k×w[i]]+k×v[i])
     *
     * 这里 k >= 0。
     * *****************************************************************************************************************
     * 参考代码 1：
     * 复杂度分析：
     * 时间复杂度：O(NW^2) ，这里 N 是背包价值数组的长度，W 是背包的容量；
     * 空间复杂度：O(NW)。
     * *****************************************************************************************************************
     * 说明：这一版代码的时间复杂度很高，使用了三重循环，有重复计算。
     * 优化「状态转移方程」（重点）
     *
     * 注意：下面这部分内容可能有一些繁琐，如果阅读有困难建议读者在纸上手写推导。
     * 状态定义：dp[i][j] = max(dp[i - 1][j - k · w[i]] + k · v[i])，这里 k >= 0。 ①
     * 单独把 k = 0 拿出来，作为一个 max 的比较项。
     * dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - k · w[i]] + k · v[i])，这里 k >= 1。 ②
     * 而当 k >= 1 的时候，把 v[i] 单独拿出来。
     * max(dp[i - 1][j - k · w[i]] + k · v[i]) = v[i] + max(dp[i - 1][j - k · w[i]] + (k - 1) · v[i]) ③
     * 将 ① 中左边的 j 用 j - w[k] 代入：
     * dp[i][j - w[i]] = max(dp[i - 1][j - w[i] - k · w[i]] + k · v[i])，这里 k >= 0。
     * = max(dp[i - 1][j - (k + 1)· w[i]] + k · v[i])，这里 k >= 0。
     * = max(dp[i - 1][j - k· w[i]] + (k - 1) · v[i])，这里 k >= 1。④
     * 结合 ②、③ 和 ④，推出 dp[i][j] = max(dp[i - 1][j], dp[i][j - w[i]]) + v[i]。
     * 比较「0-1 背包」与「完全背包」问题的状态转移方程（重点）
     * 关键在于「填表顺序」。
     * 区别只在红色标出来的地方：「0 - 1」背包参考上一行，「完全背包」参考当前行。所以优化空间的写法，在一维数组里，「0 - 1」背包 倒序填表，完全背包正序填表。
     * *****************************************************************************************************************
     * 参考代码 2：使用优化的状态转移方程（二维数组）
     * 复杂度分析：
     *
     * 时间复杂度：O(NW)，这里 N 是背包价值数组的长度，W 是背包的容量；
     * 空间复杂度：O(NW)。
     * *****************************************************************************************************************
     * 参考代码 3：使用优化的状态转移方程 + 优化空间（一维数组）
     * 复杂度分析：
     *
     * 时间复杂度：O(NW)，这里 N 是背包价值数组的长度，W 是背包的容量；
     * 空间复杂度：O(N)。
     * *****************************************************************************************************************
     * 「背包问题」我们就为大家介绍到这里，「背包问题」还有很复杂的变种问题和扩展问题，已经不在通常的面试和笔试的考察范围内，如果大家感兴趣，可以在互联网上搜索相关资料。
     *
     * 练习
     * 「力扣」上的「完全背包」问题如下：
     * 1.「力扣」第 322 题：零钱兑换：三种写法：① 直接推状态转移方程；② BFS；③ 套完全背包模式（内外层顺序正好相反）；
     * 2.「力扣」第 518 题：零钱兑换 II：建议直接推公式，做等价转换。
     * 注意：「力扣」第 377 题：组合总和 Ⅳ，不是完全背包问题。
     *
     * （这部分内容待添加。）
     *
     * 总结
     * 1.「动态规划」问题是非常大的一类问题，而且也是面试和笔试过程中的常见问题和常考问题；
     * 2.「动态规划」问题刚接触的时候，可以考虑「自顶向下」「递归 + 记忆化」，熟悉了以后，考虑「自底向上」递推完成；
     * 3.但不要忽略「自顶向下」「递归 + 记忆化」的作用，有个别问题得这样求解；
     * 4.掌握常见的「动态规划」问题，理解「状态设计」的原因，最好能用自己的话说一遍，越自然越好；
     * 5.「动态规划」的问题很多，需要多做总结，但同时也要把握难度，很难、技巧很强的问题，可以暂时不掌握。
     * 参考资料
     * 互联网上搜索「背包九讲」；
     * 《挑战程序设计竞赛（第 2 版）》人民邮电出版社 第 2.3 节 《记录结果再利用的「动态规划」》，本文《优化「状态转移方程」》参考自这里，做了更细致的拆分。
     * *****************************************************************************************************************
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/coin-change/solution/dong-tai-gui-hua-shi-yong-wan-quan-bei-bao-wen-ti-/
     */
    public void packageProblem() {
        Scanner scanner = new Scanner(System.in);

        // 读第 1 行
        int N = scanner.nextInt();
        int V = scanner.nextInt();

        // 读后面的体积和价值
        int[] weight = new int[N];
        int[] value = new int[N];

        for (int i = 0; i < N; i++) {
            weight[i] = scanner.nextInt();
            value[i] = scanner.nextInt();
        }

        // dp[i][j] 表示考虑物品区间 [0, i] 里，不超过背包容量，能够获得的最大价值；
        // 因为包含价值为 0 的计算，所以 + 1
        int[][] dp = new int[N][V + 1];
        // 先写第 1 行
        for (int k = 0; k * weight[0] <= V; k++) {
            dp[0][k * weight[0]] = k * value[0];
        }

        // 最朴素的做法
        for (int i = 1; i < N; i++) {
            for (int j = 0; j <= V; j++) {
                // 多一个 for 循环，枚举下标为 i 的物品可以选的个数
                for (int k = 0; k * weight[i] <= j; k++) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - k * weight[i]] + k * value[i]);
                }
            }
        }
        // 输出
        System.out.println(dp[N - 1][V]);
    }

    /**
     * 参考代码 2：使用优化的状态转移方程（二维数组）
     * 复杂度分析：
     * 时间复杂度：O(NW)，这里 N 是背包价值数组的长度，W 是背包的容量；
     * 空间复杂度：O(NW)。
     */
    public void packageProblem2() {
        Scanner scanner = new Scanner(System.in);

        // 读第 1 行
        int N = scanner.nextInt();
        int V = scanner.nextInt();

        // 读后面的体积和价值
        int[] weight = new int[N];
        int[] value = new int[N];

        for (int i = 0; i < N; i++) {
            weight[i] = scanner.nextInt();
            value[i] = scanner.nextInt();
        }

        // dp[i][j] 表示考虑物品区间 [0, i] 里，不超过背包容量，能够获得的最大价值；
        // 因为包含价值为 0 的计算，所以 + 1
        int[][] dp = new int[N + 1][V + 1];
        // 优化
        for (int i = 1; i <= N; i++) {
            for (int j = 0; j <= V; j++) {
                // 至少是上一行抄下来
                dp[i][j] = dp[i - 1][j];
                if (weight[i - 1] <= j) {
                    dp[i][j] = Math.max(dp[i][j], dp[i][j - weight[i - 1]] + value[i - 1]);
                }
            }
        }
        // 输出
        System.out.println(dp[N][V]);
    }

    /**
     * 参考代码 3：使用优化的状态转移方程 + 优化空间（一维数组）
     * 复杂度分析：
     * 时间复杂度：O(NW)，这里 N 是背包价值数组的长度，W 是背包的容量；
     * 空间复杂度：O(N)。
     */
    public void packageProblem3() {
        Scanner scanner = new Scanner(System.in);

        // 读第 1 行
        int N = scanner.nextInt();
        int V = scanner.nextInt();

        // 读后面的体积和价值
        int[] weight = new int[N];
        int[] value = new int[N];

        for (int i = 0; i < N; i++) {
            weight[i] = scanner.nextInt();
            value[i] = scanner.nextInt();
        }

        int[] dp = new int[V + 1];
        // 先写第 1 行

        // 优化空间
        for (int i = 1; i <= N; i++) {
            // 细节，j 从 weight[i - 1] 开始遍历
            for (int j = weight[i - 1]; j <= V; j++) {
                dp[j] = Math.max(dp[j], dp[j - weight[i - 1]] + value[i - 1]);
            }
        }
        // 输出
        System.out.println(dp[V]);
    }

    /**
     * 198. 打家劫舍（字节跳动、谷歌、亚马逊在半年内面试中考过）
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
     * *****************************************************************************************************************
     * 输入：[1,2,3,1]
     * 输出：4
     * 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     * 偷窃到的最高金额 = 1 + 3 = 4 。
     * *****************************************************************************************************************
     * 方法1:超哥的方法
     * 0 不偷 ; 1 偷
     */
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int n = nums.length;
        int[][] a = new int[n][2];
        a[0][0] = 0;
        a[0][1] = nums[0];
        for (int i = 1; i < n; i++) {
            a[i][0] = Math.max(a[i - 1][0], a[i - 1][1]);
            a[i][1] = a[i - 1][0] + nums[i];
        }
        return Math.max(a[n - 1][0], a[n - 1][1]);
    }

    /**
     * dp[i] : 0..i 能偷到的 max value , 第i个房子可偷可不偷;
     */
    public int rob2(int[] nums) {
        int n = nums.length;
        if (n <= 1) return n == 0 ? 0 : nums[0];
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[n - 1];
    }

    /**
     * python 解法
     * def rob(self, nums):
     *         """
     *         :type nums: List[int]
     *         :rtype: int
     *         """
     *         pre = 0;
     *         now = 0;
     *         for i in nums:
     *             pre , now = now ,max(pre + i ,now);
     *         return now;
     */

    /**
     * 213. 打家劫舍 II
     * 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。
     * 同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额。
     * *****************************************************************************************************************
     * 输入：nums = [2,3,2]
     * 输出：3
     * 解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
     * *****************************************************************************************************************
     * 打家劫舍 II（动态规划，结构化思路，清晰题解）
     * 解题思路：
     * 总体思路：
     * 此题是 198. 打家劫舍 的拓展版： 唯一的区别是此题中的房间是环状排列的（即首尾相接），而 198. 题中的房间是单排排列的；而这也是此题的难点。
     * 环状排列意味着第一个房子和最后一个房子中只能选择一个偷窃，因此可以把此环状排列房间问题约化为两个单排排列房间子问题：
     * 在不偷窃第一个房子的情况下（即 nums[1:]），最大金额是 p1
     * 在不偷窃最后一个房子的情况下（即 nums[:n−1]），最大金额是 p2
     * 综合偷窃最大金额： 为以上两种情况的较大值，即 max(p1,p2) 。
     * 下面的任务则是解决 单排排列房间（即 198. 打家劫舍） 问题。推荐可以先把 198. 做完再做这道题。
     * *****************************************************************************************************************
     * 198. 解题思路：
     * 典型的动态规划，以下按照标准流程解题。
     * 一.状态定义：
     * 设动态规划列表 dp ，dp[i] 代表前 i 个房子在满足条件下的能偷窃到的最高金额。
     * 二.转移方程：
     * 设： 有 n 个房子，前 n 间能偷窃到的最高金额是 dp[n] ，前 n−1 间能偷窃到的最高金额是 dp[n−1] ，此时向这些房子后加一间房，此房间价值为 num ；
     * 加一间房间后： 由于不能抢相邻的房子，意味着抢第 n+1 间就不能抢第 n 间；那么前 n+1 间房能偷取到的最高金额 dp[n+1] 一定是以下两种情况的 较大值 ：
     * 1.不抢第 n+1 个房间，因此等于前 n 个房子的最高金额，即 dp[n+1] = dp[n] ；
     * 2.抢第 n+1 个房间，此时不能抢第 n 个房间；因此等于前 n−1 个房子的最高金额加上当前房间价值，即 dp[n+1] = dp[n-1] + num ；
     * *****************************************************************************************************************
     * 细心的我们发现： 难道在前 n 间的最高金额 dp[n] 情况下，第 n 间一定被偷了吗？假设没有被偷，那 n+1 间的最大值应该也可能是 dp[n+1] = dp[n] + num 吧？其实这种假设的情况可以被省略，这是因为：
     * 假设第 n 间没有被偷，那么此时 dp[n] = dp[n-1] ，此时 dp[n+1] = dp[n] + num = dp[n-1] + num ，即可以将 两种情况合并为一种情况 考虑；
     * 假设第 n 间被偷，那么此时 dp[n+1] = dp[n] + num 不可取 ，因为偷了第 n 间就不能偷第 n+1 间。
     * 最终的转移方程： dp[n+1] = max(dp[n],dp[n-1]+num)
     * 三.初始状态：
     * 前 0 间房子的最大偷窃价值为 0 ，即 dp[0] = 0 。
     * 四.返回值：
     * 返回 dp 列表最后一个元素值，即所有房间的最大偷窃价值。
     * 五.简化空间复杂度：
     * 我们发现 dp[n] 只与 dp[n-1] 和 dp[n-2] 有关系，因此我们可以设两个变量 cur和 pre 交替记录，将空间复杂度降到 O(1) 。
     * 复杂度分析：
     * 时间复杂度 O(N) ： 两次遍历 nums 需要线性时间；
     * 空间复杂度 O(1) ： cur和 pre 使用常数大小的额外空间。
     * 链接：https://leetcode-cn.com/problems/house-robber-ii/solution/213-da-jia-jie-she-iidong-tai-gui-hua-jie-gou-hua-/
     * *****************************************************************************************************************
     * 问题转化不太严谨的，因为对于一个环来说，如果求最大值，存在首尾两个节点都不取的情况；
     * 但为什么问题可以转化为求两个队列呢？
     * 因为对于上述情况，即首尾都不取时，它的最大值肯定小于等于只去掉首或者只去掉尾的队列。
     * 即f（n1,n2,n3）<=f(n1,n2,n3,n4)
     * *****************************************************************************************************************
     * 感谢补充，非常清晰！但其实，本文的假设是第一个房子和最后一个房子中只能选择一个偷窃，这个等价于两个房子不能同时偷窃，即并没有排除首尾两个节点都不取的情况。
     *（例如，在假设不偷第一个房子时，转化为nums[1:]，在此情况下，不偷最后一个房子的情况仍然会计算进去）。
     * *****************************************************************************************************************
     * 其实就是把环拆成两个队列，一个是从0到n-1，另一个是从1到n，然后返回两个结果最大的。
     * Arrays.copyOfRange(int[] original, int from, int to)
     * param original   the array from which a range is to be copied
     * param from       the initial index of the range to be copied, inclusive
     * param to         the final index of the range to be copied, exclusive.
     * from----inclusive ; to---exclusive .
     * 将一个原始的数组original，从下标from开始复制，复制到上标to，生成一个新的数组。
     * 注意这里包括下标from，不包括上标to。
     * *****************************************************************************************************************
     */
    public int rob3(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        return Math.max(myRob(Arrays.copyOfRange(nums, 0, nums.length - 1)), myRob(Arrays.copyOfRange(nums, 1, nums.length)));
    }

    private int myRob(int[] nums) {
        int pre = 0, cur = 0, tmp;
        for (int num : nums) {
            tmp = cur;
            cur = Math.max(pre + num, cur);
            pre = tmp;
        }
        return cur;
    }

    /**
     * 213. House Robber II
     * You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle.
     * That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.
     * Given a list of non-negative integers nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.
     * Example 1:
     * Input: nums = [2,3,2]
     * Output: 3
     * Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
     * Example 2:
     * Input: nums = [1,2,3,1]
     * Output: 4
     * Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
     * Total amount you can rob = 1 + 3 = 4.
     * *****************************************************************************************************************
     * https://leetcode.com/problems/house-robber-ii/discuss/1006599/Java-or-beats-100-ile-or-easy-to-understand
     * Java | beats 100%-ile | easy to understand
     */
    public int rob(int[] nums, int start, int end) {
        int included = 0, excluded = 0;

        for (int index = start; index <= end; index++) {
            int i = included;
            int e = excluded;

            included = nums[index] + e;
            excluded = Math.max(i, e);
        }

        return Math.max(included, excluded);
    }

    public int rob4(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int n = nums.length;
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[0], nums[1]);
        return Math.max(rob(nums, 0, n - 2), rob(nums, 1, n - 1));
    }

    /**
     * python 解法
     * def rob(self, nums: [int]) -> int:
     *         def my_rob(nums):
     *             cur, pre = 0, 0
     *             for num in nums:
     *                 cur, pre = max(pre + num, cur), cur
     *             return cur
     *         return max(my_rob(nums[:-1]),my_rob(nums[1:])) if len(nums) != 1 else nums[0]
     *
     * 链接：https://leetcode-cn.com/problems/house-robber-ii/solution/213-da-jia-jie-she-iidong-tai-gui-hua-jie-gou-hua-/
     */

    /**
     * 121. 买卖股票的最佳时机（亚马逊、字节跳动、Facebook 在半年内面试中常考）
     * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     * 如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
     * 注意：你不能在买入股票前卖出股票。
     * *****************************************************************************************************************
     * 示例 1:
     * 输入: [7,1,5,3,6,4]
     * 输出: 5
     * 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     *      注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
     * 示例 2:
     * 输入: [7,6,4,3,1]
     * 输出: 0
     * 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock
     * *****************************************************************************************************************
     * 方法一：暴力法
     */
    public int maxProfit(int[] prices) {
        int maxprofit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            for (int j = i + 1; j < prices.length; j++) {
                int profit = prices[j] - prices[i];
                if (profit > maxprofit) {
                    maxprofit = profit;
                }
            }
        }
        return maxprofit;
    }
    /**
     * 方法二：一次遍历
     * 复杂度分析
     * 时间复杂度：O(n)，只需要遍历一次。
     * 空间复杂度：O(1)，只使用了常数个变量。
     */
    public int maxProfit2(int[] prices) {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minprice) {
                minprice = prices[i];
            } else if (prices[i] - minprice > maxprofit) {
                maxprofit = prices[i] - minprice;
            }
        }
        return maxprofit;
    }

    /**
     * 暴力解法、动态规划（Java）
     * 本题解于 2020 年 10 月 21 日重写；
     * 推荐阅读 @stormsunshine 编写的文章《股票问题系列通解（转载翻译）》；
     * 这一题的知识点：
     * 1.动态规划用于求解 多阶段决策问题 ；
     * 2.动态规划问题的问法：只问最优解，不问具体的解；
     * 3.掌握 无后效性 解决动态规划问题：把约束条件设置成为状态。
     * 这一系列问题的目录：
     * *****************************************************************************************************************
     * 题号	题解
     * 121. 买卖股票的最佳时机	暴力解法、动态规划（Java）
     * 122. 买卖股票的最佳时机 II	暴力搜索、贪心算法、动态规划（Java）
     * 123. 买卖股票的最佳时机 III	动态规划（Java）
     * 188. 买卖股票的最佳时机 IV	动态规划（「力扣」更新过用例，只有优化空间的版本可以 AC）
     * 309. 最佳买卖股票时机含冷冻期	动态规划（Java）
     * 714. 买卖股票的最佳时机含手续费	动态规划（Java）
     * *****************************************************************************************************************
     * 方法一：暴力解法
     * 思路：枚举所有发生一次交易的股价差。
     * 参考代码 1：
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/bao-li-mei-ju-dong-tai-gui-hua-chai-fen-si-xiang-b/
     */

    public int maxProfit3(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }

        // 有可能不发生交易，因此结果集的初始值设置为 0
        int res = 0;

        // 枚举所有发生一次交易的股价差
        for (int i = 0; i < len - 1; i++) {
            for (int j = i + 1; j < len; j++) {
                res = Math.max(res, prices[j] - prices[i]);
            }
        }
        return res;
    }
    /**
     * 方法二：动态规划
     * 思路：题目只问最大利润，没有问这几天具体哪一天买、哪一天卖，因此可以考虑使用 动态规划 的方法来解决。
     * 买卖股票有约束，根据题目意思，有以下两个约束条件：
     *
     * 条件 1：你不能在买入股票前卖出股票；
     * 条件 2：最多只允许完成一笔交易。
     * 因此 当天是否持股 是一个很重要的因素，而当前是否持股和昨天是否持股有关系，为此我们需要把 是否持股 设计到状态数组中。
     *
     * 状态定义：
     *
     * dp[i][j]：下标为 i 这一天结束的时候，手上持股状态为 j 时，我们持有的现金数。
     *
     * j = 0，表示当前不持股；
     * j = 1，表示当前持股。
     * 注意：这个状态具有前缀性质，下标为 i 的这一天的计算结果包含了区间 [0, i] 所有的信息，因此最后输出 dp[len - 1][0]。
     *
     * 说明：
     *
     * 使用「现金数」这个说法主要是为了体现 买入股票手上的现金数减少，卖出股票手上的现金数增加 这个事实；
     * 「现金数」等价于题目中说的「利润」，即先买入这只股票，后买入这只股票的差价；
     * 因此在刚开始的时候，我们的手上肯定是有一定现金数能够买入这只股票，即刚开始的时候现金数肯定不为 00，但是写代码的时候可以设置为 0。极端情况下（股价数组为 [5, 4, 3, 2, 1]），此时不发生交易是最好的（这一点是补充说明，限于我的表达，希望不要给大家造成迷惑）。
     * 推导状态转移方程：
     *
     * dp[i][0]：规定了今天不持股，有以下两种情况：
     *
     * 昨天不持股，今天什么都不做；
     * 昨天持股，今天卖出股票（现金数增加），
     * dp[i][1]：规定了今天持股，有以下两种情况：
     *
     * 昨天持股，今天什么都不做（现金数增加）；
     * 昨天不持股，今天买入股票（注意：只允许交易一次，因此手上的现金数就是当天的股价的相反数）。
     * 状态转移方程请见 参考代码 2。
     *
     * 知识点：
     *
     * 多阶段决策问题：动态规划常常用于求解多阶段决策问题；
     * 无后效性：每一天是否持股设计成状态变量的一维。状态设置具体，推导状态转移方程方便。
     * 参考代码 2：
     *
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/bao-li-mei-ju-dong-tai-gui-hua-chai-fen-si-xiang-b/
     * 来源：力扣（LeetCode）
     * 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
     */
    public int maxProfit4(int[] prices) {
        int len = prices.length;
        // 特殊判断
        if (len < 2) {
            return 0;
        }
        int[][] dp = new int[len][2];

        // dp[i][0] 下标为 i 这天结束的时候，不持股，手上拥有的现金数
        // dp[i][1] 下标为 i 这天结束的时候，持股，手上拥有的现金数

        // 初始化：不持股显然为 0，持股就需要减去第 1 天（下标为 0）的股价
        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        // 从第 2 天开始遍历
        for (int i = 1; i < len; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
        }
        return dp[len - 1][0];
    }

    /**
     * 122. 买卖股票的最佳时机 II
     * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     * 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     * *****************************************************************************************************************
     * 暴力搜索、贪心算法、动态规划（Java）
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/solution/tan-xin-suan-fa-by-liweiwei1419-2/
     * 推荐阅读 @stormsunshine 编写的文章《股票问题系列通解（转载翻译）》
     * 链接：https://leetcode-cn.com/circle/article/qiAgHn/
     * *****************************************************************************************************************
     * 输入: [7,1,5,3,6,4]
     * 输出: 7
     * 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     *      随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
     * *****************************************************************************************************************
     * 示例 2:
     * 输入: [1,2,3,4,5]
     * 输出: 4
     * 解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     *      注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     *      因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
     * 示例3:
     * 输入: [7,6,4,3,1]
     * 输出: 0
     * 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
     * *****************************************************************************************************************
     * 方法一：暴力搜索（超时）
     * 根据题意：由于不限制交易次数，在每一天，就可以根据当前是否持有股票选择相应的操作。「暴力搜索」在树形问题里也叫「回溯搜索」、「回溯法」。
     * 因为超时所以不采取以下这种方式
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/solution/tan-xin-suan-fa-by-liweiwei1419-2/
     */
    private int res;

    public int maxProfit21(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }
        this.res = 0;
        dfs(prices, 0, len, 0, res);
        return this.res;
    }

    /**
     * @param prices 股价数组
     * @param index  当前是第几天，从 0 开始
     * @param status 0 表示不持有股票，1表示持有股票，
     * @param profit 当前收益
     */
    private void dfs(int[] prices, int index, int len, int status, int profit) {
        if (index == len) {
            this.res = Math.max(this.res, profit);
            return;
        }
        dfs(prices, index + 1, len, status, profit);
        if (status == 0) {
            // 可以尝试转向 1
            dfs(prices, index + 1, len, 1, profit - prices[index]);
        } else {
            // 此时 status == 1，可以尝试转向 0
            dfs(prices, index + 1, len, 0, profit + prices[index]);
        }
    }

    /**
     * 方法二：动态规划（通用）
     * *****************************************************************************************************************
     * 根据 「力扣」第 121 题的思路，需要设置一个二维矩阵表示状态。
     * 第 1 步：定义状态
     * 状态 dp[i][j] 定义如下：
     * dp[i][j] 表示到下标为 i 的这一天，持股状态为 j 时，我们手上拥有的最大现金数。
     * 注意：限定持股状态为 j 是为了方便推导状态转移方程，这样的做法满足 无后效性。
     * 其中：
     * 第一维 i 表示下标为 i 的那一天（ 具有前缀性质，即考虑了之前天数的交易 ）；
     * 第二维 j 表示下标为 i 的那一天是持有股票，还是持有现金。这里 0 表示持有现金（cash），1 表示持有股票（stock）。
     * *****************************************************************************************************************
     * 第 2 步：思考状态转移方程
     * 状态从持有现金（cash）开始，到最后一天我们关心的状态依然是持有现金（cash）；
     * 每一天状态可以转移，也可以不动。状态转移用下图表示：
     * 如下链接所示
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/solution/tan-xin-suan-fa-by-liweiwei1419-2/
     * （状态转移方程写在代码中）
     * *****************************************************************************************************************
     * 说明：
     * 由于不限制交易次数，除了最后一天，每一天的状态可能不变化，也可能转移；
     * 写代码的时候，可以不用对最后一天单独处理，输出最后一天，状态为 0 的时候的值即可。
     * *****************************************************************************************************************
     * 第 3 步：确定初始值
     * 起始的时候：
     * 如果什么都不做，dp[0][0] = 0；
     * 如果持有股票，当前拥有的现金数是当天股价的相反数，即 dp[0][1] = -prices[i]；
     * *****************************************************************************************************************
     * 第 4 步：确定输出值
     * 终止的时候，上面也分析了，输出 dp[len - 1][0]，因为一定有 dp[len - 1][0] > dp[len - 1][1]。
     * *****************************************************************************************************************
     * 参考代码 2：
     * prices[i]代表第i天股价
     */
    public int maxProfit22(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }

        // 0：持有现金
        // 1：持有股票
        // 状态转移：0 → 1 → 0 → 1 → 0 → 1 → 0
        int[][] dp = new int[len][2];

        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < len; i++) {
            // 这两行调换顺序也是可以的
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[len - 1][0];
    }

    /**
     * 先说这行：
     * dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
     * dp[i-1][0]代表的是我昨天手里的现金，
     * dp[i-1][1]+prices[i]代表的是如果我今天卖掉手里的现金，
     * 如果我今天卖掉，我现在手里的现金比昨天多，那我就卖掉，反之我就不卖；
     * 这行：
     * dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
     * dp[i-1][1]代表的是我昨天持有的股票价值
     * dp[i-1][0]-prices[i]代表的是如果我今天买股票我手头会有的股票价值
     * 所以，如果我今天买股票，我手头的股票价值会比昨天高，那我就买，反之我就不买
     * 这道题的核心思想就是低价买，高价卖，其实也可以直接遍历一遍看如果今天的股票价格比昨天高，那我就加上这个差价，如果比昨天低，那我就什么也不干。也是可以的。
     * int maxProfit = 0
     * for (int = 1; i<len; i++) {
     *  maxProfit = maxProfit + Math.max(0, prices[i] - princes[i-1])
     * }
     * return maxProfit
     * *****************************************************************************************************************
     * 我们也可以将状态数组分开设置。
     * 参考代码 3：
     */
    public int maxProfit23(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }

        // cash：持有现金
        // hold：持有股票
        // 状态数组
        // 状态转移：cash → hold → cash → hold → cash → hold → cash
        int[] cash = new int[len];
        int[] hold = new int[len];

        cash[0] = 0;
        hold[0] = -prices[0];

        for (int i = 1; i < len; i++) {
            // 这两行调换顺序也是可以的
            cash[i] = Math.max(cash[i - 1], hold[i - 1] + prices[i]);
            hold[i] = Math.max(hold[i - 1], cash[i - 1] - prices[i]);
        }
        return cash[len - 1];
    }

    /**
     * 参考代码 6：
     * https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/solution/tan-xin-suan-fa-by-liweiwei1419-2/
     */
    public int maxProfit26(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }

        int maxProfit = 0;
        for (int i = 1; i < len; i++) {
            maxProfit = maxProfit + Math.max(0, prices[i] - prices[i - 1]);
        }
        return maxProfit;
    }

    /**
     * The greedy pair-wise approach mentioned in other posts is great for this problem indeed, but if we're not allowed to buy and sell stocks within the same day it can't be applied (logically, of course; the answer will be the same).
     * Actually, the straight-forward way of finding next local minimum and next local maximum is not much more complicated, so, just for the sake of having an alternative I share the code in Java for such case.
     */
    public int maxProfit27(int[] prices) {
        int profit = 0, i = 0;
        while (i < prices.length) {
            // find next local minimum
            while (i < prices.length - 1 && prices[i + 1] <= prices[i]) i++;
            int min = prices[i++]; // need increment to avoid infinite loop for "[1]"
            // find next local maximum
            while (i < prices.length - 1 && prices[i + 1] >= prices[i]) i++;
            profit += i < prices.length ? prices[i++] - min : 0;
        }
        return profit;
    }

    /**
     * The profit is the sum of sub-profits. Each sub-profit is the difference between selling at day j, and buying at day i (with j > i). The range [i, j] should be chosen so that the sub-profit is maximum:
     * sub-profit = prices[j] - prices[i]
     * We should choose j that prices[j] is as big as possible, and choose i that prices[i] is as small as possible (of course in their local range).
     * Let's say, we have a range [3, 2, 5], we will choose (2,5) instead of (3,5), because 2<3.
     * Now, if we add 8 into this range: [3, 2, 5, 8], we will choose (2, 8) instead of (2,5) because 8>5.
     * From this observation, from day X, the buying day will be the last continuous day that the price is smallest. Then, the selling day will be the last continuous day that the price is biggest.
     * Take another range [3, 2, 5, 8, 1, 9], though 1 is the smallest, but 2 is chosen, because 2 is the smallest in a consecutive decreasing prices starting from 3.
     * Similarly, 9 is the biggest, but 8 is chosen, because 8 is the biggest in a consecutive increasing prices starting from 2 (the buying price).
     * Actually, the range [3, 2, 5, 8, 1, 9] will be splitted into 2 sub-ranges [3, 2, 5, 8] and [1, 9].
     * We come up with the following code:
     * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/discuss/208241/Explanation-for-the-dummy-like-me.
     */
    public int maxProfit28(int[] prices) {
        int i = 0, buy, sell, profit = 0, N = prices.length - 1;
        while (i < N) {
            while (i < N && prices[i + 1] <= prices[i]) i++;
            buy = prices[i];

            while (i < N && prices[i + 1] > prices[i]) i++;
            sell = prices[i];

            profit += sell - buy;
        }
        return profit;
    }

    /**
     * BONUS:
     * With this approach, we still can calculate on which buying days and selling days, we reach the max profit:
     */
    public Pair<List<Pair<Integer, Integer>>, Integer> buysAndSells(int[] prices) {
        int i = 0, iBuy, iSell, profit = 0, N = prices.length - 1;
        List<Pair<Integer, Integer>> buysAndSells = new ArrayList<Pair<Integer, Integer>>();
        while (i < N) {
            while (i < N && prices[i + 1] <= prices[i]) i++;
            iBuy = i;

            while (i < N && prices[i + 1] > prices[i]) i++;
            iSell = i;

            profit += prices[iSell] - prices[iBuy];
            Pair<Integer, Integer> bs = new Pair<Integer, Integer>(iBuy, iSell);
            buysAndSells.add(bs);
        }
        return new Pair<List<Pair<Integer, Integer>>, Integer>(buysAndSells, profit);
    }

    public class Pair<T1, T2> {
        public T1 fst;
        public T2 snd;

        public Pair(T1 f, T2 s) {
            fst = f;
            snd = s;
        }
    }


    /**
     * 123. 买卖股票的最佳时机 III （字节跳动在半年内面试中考过）
     * 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
     * 设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii
     * *****************************************************************************************************************
     * 示例 1:
     * 输入：prices = [3,3,5,0,0,3,1,4]
     * 输出：6
     * 解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     *      随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
     * 示例 2：
     * 输入：prices = [1,2,3,4,5]
     * 输出：4
     * 解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     *      注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     *      因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
     * 示例 3：
     * 输入：prices = [7,6,4,3,1]
     * 输出：0
     * 解释：在这个情况下, 没有交易完成, 所以最大利润为 0。
     * *****************************************************************************************************************
     * 123. 买卖股票的最佳时机 III【动态规划经典】详解
     * 一.思路
     * 这道题目相对 121.买卖股票的最佳时机 和 122.买卖股票的最佳时机II 难了不少。
     * 关键在于至多买卖两次，这意味着可以买卖一次，可以买卖两次，也可以不买卖。
     * 接来下我用动态规划五部曲详细分析一下：
     * *****************************************************************************************************************
     * 二.确定dp数组以及下标的含义
     * 一天一共就有五个状态，
     * 0. 没有操作
     * 1. 第一次买入
     * 2. 第一次卖出
     * 3. 第二次买入
     * 4. 第二次卖出
     * dp[i][j]中 i表示第i天，j为 [0 - 4] 五个状态，dp[i][j]表示第i天状态j所剩最大现金。
     * *****************************************************************************************************************
     * 三.确定递推公式
     * dp[i][0] = dp[i - 1][0];
     * 需要注意：dp[i][1]，表示的是第i天，买入股票的状态，并不是说一定要第i天买入股票，这是很多同学容易陷入的误区。
     * 达到dp[i][1]状态，有两个具体操作：
     * 操作一：第i天买入股票了，那么dp[i][1] = dp[i-1][0] - prices[i]
     * 操作二：第i天没有操作，而是沿用前一天买入的状态，即：dp[i][1] = dp[i - 1][1]
     * 那么dp[i][1]究竟选 dp[i-1][0] - prices[i]，还是dp[i - 1][1]呢？
     * 一定是选最大的，所以 dp[i][1] = max(dp[i-1][0] - prices[i], dp[i - 1][1]);
     * 同理dp[i][2]也有两个操作：
     * 操作一：第i天卖出股票了，那么dp[i][2] = dp[i - 1][1] + prices[i]
     * 操作二：第i天没有操作，沿用前一天卖出股票的状态，即：dp[i][2] = dp[i - 1][2]
     * 所以dp[i][2] = max(dp[i - 1][1] + prices[i], dp[i - 1][2])
     * 同理可推出剩下状态部分：
     * dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
     * dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
     * *****************************************************************************************************************
     * 四.dp数组如何初始化
     * 第0天没有操作，这个最容易想到，就是0，即：dp[0][0] = 0;
     * 第0天做第一次买入的操作，dp[0][1] = -prices[0];
     * 第0天做第一次卖出的操作，这个初始值应该是多少呢？
     * 首先卖出的操作一定是收获利润，整个股票买卖最差情况也就是没有盈利即全程无操作现金为0，
     * 从递推公式中可以看出每次是取最大值，那么既然是收获利润如果比0还小了就没有必要收获这个利润了。
     * 所以dp[0][2] = 0;
     * 第0天第二次买入操作，初始值应该是多少呢？
     * 不用管第几次，现在手头上没有现金，只要买入，现金就做相应的减少。
     * 所以第二次买入操作，初始化为：dp[0][3] = -prices[0];
     * 同理第二次卖出初始化dp[0][4] = 0;
     * PS：相信很多小伙伴刷题的时候面对力扣上近两千到题目，感觉无从下手，我花费半年时间整理了leetcode刷题攻略。 里面有100多道经典算法题目刷题顺序、配有40w字的详细图解，常用算法模板总结，以及难点视频题解，按照list一道一道刷就可以了！star支持一波吧！
     * *****************************************************************************************************************
     * 五.确定遍历顺序
     * 从递归公式其实已经可以看出，一定是从前向后遍历，因为dp[i]，依靠dp[i - 1]的数值。
     * *****************************************************************************************************************
     * 六.举例推导dp数组
     * 以输入[1,2,3,4,5]为例
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/solution/123-mai-mai-gu-piao-de-zui-jia-shi-ji-ii-zfh9/
     * 大家可以看到红色框为最后两次卖出的状态。
     * 现在最大的时候一定是卖出的状态，而两次卖出的状态现金最大一定是最后一次卖出。
     * 所以最终最大利润是 dp[4][4]
     * 时间复杂度：O(n)
     * 空间复杂度：O(n * 5)
     * 当然，大家在网上看到的有的题解还有一种优化空间写法，如下：
     * 但这种写法，dp[2] 利用的是当天的dp[1]，如果强行解释理解可以把dp数组打印出来，确实是答案，但从理论推导感觉说不通，可能这就是神代码吧，欢迎大家来讨论一波！
     * 「代码随想录」目前正在循序渐进讲解算法，目前已经讲到了动态规划，点击这里和上万录友一起打卡学习，来看看，你一定会发现相见恨晚！
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/solution/123-mai-mai-gu-piao-de-zui-jia-shi-ji-ii-zfh9/
     */
    public int maxProfit31(int[] prices) {
        if (prices.length == 0) return 0;
        int[][] dp = new int[prices.length][5];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        dp[0][3] = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = dp[i - 1][0];
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
        }
        return dp[prices.length - 1][4];
    }

    public int maxProfit32(int[] prices) {
        if (prices.length == 0) return 0;
        int[] dp = new int[5];
        dp[0] = 0;
        dp[1] = -prices[0];
        dp[3] = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            dp[1] = Math.max(dp[1], dp[0] - prices[i]);
            dp[2] = Math.max(dp[2], dp[1] + prices[i]);
            dp[3] = Math.max(dp[3], dp[2] - prices[i]);
            dp[4] = Math.max(dp[4], dp[3] + prices[i]);
        }
        return dp[4];
    }

    /**
     * 动态规划（Java）
     * 本题解于 2020 年 10 月 22 日重写；
     * 推荐阅读 @stormsunshine 编写的文章《股票问题系列通解（转载翻译）》。
     * 思路：这一题「最多可以完成两笔交易」是题目中给出的约束信息，因此 需要把已经交易了多少次设置成为一个状态的维度。
     * 难点：理解初始化的时候设置 dp[0][2][1] = 负无穷。
     * *****************************************************************************************************************
     * 这一系列问题的目录：
     * 题号	题解
     * 121. 买卖股票的最佳时机	        暴力解法、动态规划（Java）
     * 122. 买卖股票的最佳时机 II	    暴力搜索、贪心算法、动态规划（Java）
     * 123. 买卖股票的最佳时机 III	    动态规划（Java）
     * 188. 买卖股票的最佳时机 IV	    动态规划（「力扣」更新过用例，只有优化空间的版本可以 AC）
     * 309. 最佳买卖股票时机含冷冻期	    动态规划（Java）
     * 714. 买卖股票的最佳时机含手续费	动态规划（Java）
     * *****************************************************************************************************************
     * 方法一：动态规划
     * 一.状态定义：dp[i][j][k] 表示在 [0, i] 区间里（状态具有前缀性质），交易进行了 j 次，并且状态为 k 时我们拥有的现金数。其中 j 和 k 的含义如下：
     * j = 0 表示没有交易发生；
     * j = 1 表示此时已经发生了 1 次买入股票的行为；
     * j = 2 表示此时已经发生了 2 次买入股票的行为。
     * 即我们 人为规定 记录一次交易产生是在 买入股票 的时候。
     * k = 0 表示当前不持股；
     * k = 1 表示当前持股。
     * *****************************************************************************************************************
     * 二.推导状态转移方程：
     * 「状态转移方程」可以用下面的图表示，它的特点是：状态要么什么都不做，要么向后面走，即：状态不能回退。
     * 具体表示式请见代码注释。
     * *****************************************************************************************************************
     * 三.思考初始化：
     * 下标为 0 这一天，交易次数为 0、1、2 并且状态为 0 和 1 的初值应该如下设置：
     * dp[0][0][0] = 0：这是显然的；
     * dp[0][0][1]：表示一次交易都没有发生，但是持股，这是不可能的，也不会有后序的决策要用到这个状态值，可以不用管；
     * dp[0][1][0] = 0：表示发生了 1 次交易，但是不持股，这是不可能的。虽然没有意义，但是设置成 0 不会影响最优值；
     * dp[0][1][1] = -prices[0]：表示发生了一次交易，并且持股，所以我们持有的现金数就是当天股价的相反数；
     * dp[0][2][0] = 0：表示发生了 2 次交易，但是不持股，这是不可能的。虽然没有意义，但是设置成 0 不会影响最优值；
     * dp[0][2][1] = 负无穷：表示发生了 2 次交易，并且持股，这是不可能的。注意：虽然没有意义，但是不能设置成 0，这是因为交易还没有发生，必须规定当天 k 状态为 1（持股），需要参考以往的状态转移，一种很有可能的情况是没有交易是最好的情况。
     * 说明：dp[0][2][1] 设置成为负无穷这件事情我可能没有说清楚。大家可以通过特殊测试用例 [1, 2, 3, 4, 5]，对比 dp[0][2][1] = 0 与 dp[0][2][1] = 负无穷 的状态转移的差异去理解。
     * 注意：只有在之前的状态有被赋值的时候，才可能有当前状态。
     * 思考输出：最后一天不持股的状态都可能成为最大利润。
     * *****************************************************************************************************************
     * 参考代码 1：
     * 时间复杂度：O(N)，这里 N 表示股价数组的长度；
     * 空间复杂度：O(N)，虽然是三维数组，但是第二维、第三维是常数，与问题规模无关。
     * 以下是空间优化的代码。
     *
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/solution/dong-tai-gui-hua-by-liweiwei1419-7/
     * 来源：力扣（LeetCode）
     * 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
     */
    public int maxProfit33(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }

        // 第 2 维的 0 没有意义，1 表示交易进行了 1 次，2 表示交易进行了 2 次
        // 为了使得第 2 维的数值 1 和 2 有意义，这里将第 2 维的长度设置为 3
        int[][][] dp = new int[len][3][2];

        // 理解如下初始化
        // 第 3 维规定了必须持股，因此是 -prices[0]
        dp[0][1][1] = -prices[0];
        // 还没发生的交易，持股的时候应该初始化为负无穷
        dp[0][2][1] = Integer.MIN_VALUE;

        for (int i = 1; i < len; i++) {
            // 转移顺序先持股，再卖出
            dp[i][1][1] = Math.max(dp[i - 1][1][1], -prices[i]);
            dp[i][1][0] = Math.max(dp[i - 1][1][0], dp[i - 1][1][1] + prices[i]);
            dp[i][2][1] = Math.max(dp[i - 1][2][1], dp[i - 1][1][0] - prices[i]);
            dp[i][2][0] = Math.max(dp[i - 1][2][0], dp[i - 1][2][1] + prices[i]);
        }
        return Math.max(dp[len - 1][1][0], dp[len - 1][2][0]);
    }

    /**
     * 参考代码 2：（使用滚动数组）
     * 复杂度分析：
     * 时间复杂度：O(N)，这里 N 表示股价数组的长度；
     * 空间复杂度：O(1)，分别使用两个滚动变量，将一维数组状态优化到常数大小。
     */
    public int maxProfit34(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }

        int[][][] dp = new int[2][3][2];
        dp[0][1][1] = -prices[0];
        dp[0][2][1] = Integer.MIN_VALUE;
        for (int i = 1; i < len; i++) {
            dp[i % 2][1][1] = Math.max(dp[(i - 1) % 2][1][1], -prices[i]);
            dp[i % 2][1][0] = Math.max(dp[(i - 1) % 2][1][0], dp[(i - 1) % 2][1][1] + prices[i]);
            dp[i % 2][2][1] = Math.max(dp[(i - 1) % 2][2][1], dp[(i - 1) % 2][1][0] - prices[i]);
            dp[i % 2][2][0] = Math.max(dp[(i - 1) % 2][2][0], dp[(i - 1) % 2][2][1] + prices[i]);
        }
        return Math.max(dp[(len - 1) % 2][1][0], dp[(len - 1) % 2][2][0]);
    }

    /**
     * 参考代码 3：（由于今天只参考了昨天的状态，所以直接去掉第一维不会影响状态转移的正确性）
     * 复杂度分析：
     * 时间复杂度：O(N)，这里 N 表示股价数组的长度；
     * 空间复杂度：O(1)，状态数组的大小为常数。
     */
    public int maxProfit35(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }

        int[][] dp = new int[3][2];
        dp[1][1] = -prices[0];
        dp[2][1] = Integer.MIN_VALUE;
        for (int i = 1; i < len; i++) {
            dp[1][1] = Math.max(dp[1][1], -prices[i]);
            dp[1][0] = Math.max(dp[1][0], dp[1][1] + prices[i]);
            dp[2][1] = Math.max(dp[2][1], dp[1][0] - prices[i]);
            dp[2][0] = Math.max(dp[2][0], dp[2][1] + prices[i]);
        }
        return Math.max(dp[1][0], dp[2][0]);
    }

    /**
     * class Solution:
     *     def maxProfit(self, prices: List[int]) -> int:
     *         # 思路：这一题「最多可以完成两笔交易」是题目中给出的约束信息，因此 需要把已经交易了多少次设置成为一个状态的维度。状态定义：dp[i][j][k] 表示在 [0, i] 区间里（状态具有前缀性质），交易进行了 j 次，并且状态为 k 时我们拥有的现金数。其中 j 和 k 的含义如下：j = 0 表示没有交易发生；j = 1 表示此时已经发生了 1 次买入股票的行为；j = 2 表示此时已经发生了 2 次买入股票的行为。即我们 人为规定 记录一次交易产生是在 买入股票 的时候。k = 0 表示当前不持股；k = 1 表示当前持股。初始：dp[0][0][0] = 0：这是显然的；dp[0][1][1] = -prices[0]：表示发生了一次交易，并且持股，所以我们持有的现金数就是当天股价的相反数；dp[0][2][1] = 负无穷：表示发生了 2 次交易，并且持股，这是不可能的。优化：由于今天只参考了昨天的状态，所以直接去掉第一维不会影响状态转移的正确性。
     *         m_length = len(prices)
     *
     *         if m_length < 2:
     *             return 0
     *
     *         dp = [[0 for i in range(2)] for j in range(3)]
     *
     *         dp[1][1] = -prices[0]  # 表示发生了一次交易，并且持股，所以我们持有的现金数就是当天股价的相反数；
     *         dp[2][1] = -int(2e31)  # 表示发生了 2 次交易，并且持股，这是不可能的。为什么需要负无穷？因为不可能发生的状态应该设置为一个不可能达到的最小值. 买入时某天的手中的现金可能是负数, 如果初始化为0就会出错. 所以应该初始化为一个不会对后续转移造成影响的值
     *         for i in range(1, m_length):
     *             dp[1][1] = max(dp[1][1], -prices[i])            # 【低价入场】，让第一次入股时的价格最小。为什么是max? 因为入股的收益是负的。
     *             dp[1][0] = max(dp[1][0], dp[1][1] + prices[i])  # 【高价清仓】，当天卖出，计算收益
     *             dp[2][1] = max(dp[2][1], dp[1][0] - prices[i])  # 【再次入场】，让第二次入股时的收益最大。
     *             dp[2][0] = max(dp[2][0], dp[2][1] + prices[i])  # 当天卖出，计算收益
     *
     *         # return max(dp[1][0], dp[2][0])  # 只买一次或者买两次都可以，但是最后一天一定要清仓，计算收益。
     *         return dp[2][0]  # 最后一天一定要清仓，计算收益。
     */

    /**
     * 123. Best Time to Buy and Sell Stock III
     * Say you have an array for which the ith element is the price of a given stock on day i.
     * Design an algorithm to find the maximum profit. You may complete at most two transactions.
     * Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).
     * *****************************************************************************************************************
     * Example 1:
     * Input: prices = [3,3,5,0,0,3,1,4]
     * Output: 6
     * Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
     * Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
     * Example 2:
     * Input: prices = [1,2,3,4,5]
     * Output: 4
     * Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
     * Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.
     * *****************************************************************************************************************
     * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/39611/Is-it-Best-Solution-with-O(n)-O(1).
     * Is it Best Solution with O(n), O(1).
     * The thinking is simple and is inspired by the best solution from Single Number II (I read through the discussion after I use DP).
     * Assume we only have 0 money at first;
     * 4 Variables to maintain some interested 'ceilings' so far:
     * The maximum of if we've just buy 1st stock, if we've just sold 1nd stock, if we've just buy 2nd stock, if we've just sold 2nd stock.
     * Very simple code too and work well. I have to say the logic is simple than those in Single Number II.
     *
     */
    public int maxProfit36(int[] prices) {
        int hold1 = Integer.MIN_VALUE, hold2 = Integer.MIN_VALUE;
        int release1 = 0, release2 = 0;
        for (int i : prices) {                          // Assume we only have 0 money at first
            release2 = Math.max(release2, hold2 + i);   // The maximum if we've just sold 2nd stock so far.
            hold2 = Math.max(hold2, release1 - i);      // The maximum if we've just buy  2nd stock so far.
            release1 = Math.max(release1, hold1 + i);   // The maximum if we've just sold 1nd stock so far.
            hold1 = Math.max(hold1, -i);                // The maximum if we've just buy  1st stock so far.
        }
        return release2; // Since release1 is initiated as 0, so release2 will always higher than release1.
    }



}
