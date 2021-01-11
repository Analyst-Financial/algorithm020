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
     * <p>
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
    public int minPathSum2(int[][] grid) {
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
    public int minPathSum(int[][] grid) {
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
     * 91. 解码方法（亚马逊、Facebook、字节跳动在半年内面试中考过）
     * 一条包含字母 A-Z 的消息通过以下方式进行了编码：
     * 'A' -> 1
     * 'B' -> 2
     * ...
     * 'Z' -> 26
     * 给定一个只包含数字的非空字符串，请计算解码方法的总数。
     * 题目数据保证答案肯定是一个 32 位的整数。
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



}
