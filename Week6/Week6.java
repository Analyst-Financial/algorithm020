package algorithm;

import java.util.Arrays;

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
     */

    public int uniquePaths(int m, int n) {
        int[][] f = new int[m][n];
        for (int i = 0; i < m; ++i) {
            f[i][0] = 1;
        }
        for (int j = 0; j < n; ++j) {
            f[0][j] = 1;
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                f[i][j] = f[i - 1][j] + f[i][j - 1];
            }
        }
        return f[m - 1][n - 1];
    }

    /**
     * 超哥方法1
     */
    public int uniquePaths2(int m, int n) {
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
     */
    public int uniquePaths3(int m, int n) {
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
     * 63. 不同路径 II
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
     * 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
     * 网格中的障碍物和空位置分别用 1 和 0 来表示。
     * 输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
     * 输出：2
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
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
     * 64. 最小路径和（亚马逊、高盛集团、谷歌在半年内面试中考过）
     * 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     * 说明：每次只能向下或者向右移动一步。
     * 输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
     * 输出：7
     * 解释：因为路径 1→3→1→1→1 的总和最小。
     * https://leetcode-cn.com/problems/minimum-path-sum/solution/zui-xiao-lu-jing-he-dong-tai-gui-hua-gui-fan-liu-c/
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
}
