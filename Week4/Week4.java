package algorithm;

import java.util.*;

/**
 * @ClassName Week4
 * @Description
 * @Author Administrator
 * @Date 2020/12/8  22:32
 * @Version 1.0
 **/
public class Week4 {

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
     * 102. 二叉树的层序遍历:python
     * def levelOrder(root: TreeNode) -> List[List[int]]:
     *     if root is None:
     *         return []
     *     queue = [root]  # 把root初始化一下进去
     *     out = []
     *     while queue:
     *         child = []  # 该轮循环的结果集
     *         node = []  # 存放while下一次的数据集
     *         for item in queue:  # 把该次queue里的数据循环一下,添加到该轮循环的结果集
     *             child.append(item.val)
     *             if item.left:
     *                 node.append(item.left)  # 判断当前的数据有没有子节点，有就加到 存放while下一次的数据集
     *             if item.right:
     *                 node.append(item.right)
     *         out.append(child)  # 把while这次的结果集放到输出数组里
     *         queue = node  # 重要!!这是把node里搜集的该次循环放到queue里
     *     return out
     *
     *
     * def levelOrder2(root):
     *     nodes = [(root,)]
     *     values = []
     *     while nodes:
     *         values.append([r.val for n in nodes for r in n if r])
     *         nodes = [(r.left, r.right) for n in nodes for r in n if r]
     *     return values[:-1]
     */
    /**
     * 102. 二叉树的层序遍历
     * 给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。
     * 方法1:递归实现
     * -相同层次的节点归入同一个数组
     * -传入辅助的level参数决定层次
     */
    public List<List<Integer>> levelOrder3(TreeNode root) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        if (root == null) {
            return ret;
        }

        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<Integer>();
            int currentLevelSize = queue.size();
            for (int i = 1; i <= currentLevelSize; ++i) {
                TreeNode node = queue.poll();
                level.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            ret.add(level);
        }

        return ret;
    }

    /**
     * 102. Binary Tree Level Order Traversal (Java solution with a queue used)
     * Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
     * For example:
     * Given binary tree [3,9,20,null,null,15,7],
     * https://leetcode.com/problems/binary-tree-level-order-traversal/discuss/33450/Java-solution-with-a-queue-used
     */
    public List<List<Integer>> levelOrder4(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        List<List<Integer>> wrapList = new LinkedList<List<Integer>>();
        if (root == null) return wrapList;
        queue.offer(root);

        while (!queue.isEmpty()) {
            int levelNum = queue.size();
            List<Integer> subList = new LinkedList<Integer>();
            for (int i = 0; i < levelNum; i++) {
                if (queue.peek().left != null) queue.offer(queue.peek().left);
                if (queue.peek().right != null) queue.offer(queue.peek().right);
                subList.add(queue.poll().val);
            }
            wrapList.add(subList);
        }
        return wrapList;
    }

    /**
     * 433. Minimum Genetic Mutation
     * A gene string can be represented by an 8-character long string, with choices from "A", "C", "G", "T".
     * Suppose we need to investigate about a mutation (mutation from "start" to "end"), where ONE mutation is defined as ONE single character changed in the gene string.
     * For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation.
     * Also, there is a given gene "bank", which records all the valid gene mutations. A gene must be in the bank to make it a valid gene string.
     * Now, given 3 things - start, end, bank, your task is to determine what is the minimum number of mutations needed to mutate from "start" to "end". If there is no such a mutation, return -1.
     * Note:
     * Starting point is assumed to be valid, so it might not be included in the bank.
     * If multiple mutations are needed, all mutations during in the sequence must be valid.
     * You may assume start and end string is not the same.
     *
     * @param start 起始基因序列
     * @param end   目标基因序列
     * @param bank  基因库
     * @return
     */
    public int minMutation(String start, String end, String[] bank) {
        if (start.equals(end)) return 0;

        Set<String> bankSet = new HashSet<>();
        for (String b : bank) bankSet.add(b);

        char[] charSet = new char[]{'A', 'C', 'G', 'T'};

        int level = 0;
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.offer(start);
        visited.add(start);

        while (!queue.isEmpty()) {
            int size = queue.size();
            while (size-- > 0) {
                String curr = queue.poll();
                if (curr.equals(end)) return level;

                char[] currArray = curr.toCharArray();
                for (int i = 0; i < currArray.length; i++) {
                    char old = currArray[i];
                    for (char c : charSet) {
                        currArray[i] = c;
                        String next = new String(currArray);
                        if (!visited.contains(next) && bankSet.contains(next)) {
                            visited.add(next);
                            queue.offer(next);
                        }
                    }
                    currArray[i] = old;
                }
            }
            level++;
        }
        return -1;
    }
    /**
     * 433. 最小基因变化
     * 一条基因序列由一个带有8个字符的字符串表示，其中每个字符都属于 "A", "C", "G", "T"中的任意一个。
     * 假设我们要调查一个基因序列的变化。一次基因变化意味着这个基因序列中的一个字符发生了变化。
     * 例如，基因序列由"AACCGGTT"变化至"AACCGGTA"即发生了一次基因变化。
     * 与此同时，每一次基因变化的结果，都需要是一个合法的基因串，即该结果属于一个基因库。
     * 现在给定3个参数 start, end, bank，分别代表起始基因序列，目标基因序列及基因库，请找出能够使起始基因序列变化为目标基因序列所需的最少变化次数。如果无法实现目标变化，请返回 -1。
     * 注意:
     * 起始基因序列默认是合法的，但是它并不一定会出现在基因库中。
     * 所有的目标基因序列必须是合法的。
     * 假定起始基因序列与目标基因序列是不一样的。
     */
    /**
     * Java单向广度优先搜索和双向广度优先搜索
     * 这一题其实和单词接龙问题（单词接龙Ⅰ 、单词接龙Ⅱ）属于一类的问题，都是利用广度优先搜索解决状态图搜索问题。！！！！！！！！！
     * 这类问题的最经典就是“八数码”问题，感兴趣的朋友可以网上找找，这类问题实在太经典了，网上有很多很好的解答。这里推荐两个连接：
     * 1 八数码的八境界
     * 2 HDU 1043 八数码（八境界）
     * 以上是题目之外的东西，随便说了些，下面进入本题的题解。
     * <p>
     * 思路:单向广度优先搜索
     * 这个此类问题最基本的解法。我们先来看一幅简单的图：
     * AAAC-->AACC-->ACCC-->CCCC
     * 这个明眼人都能看出来，就是通过改每个位置上的字母来达到目标的基因排列。但是很可惜，计算机不知道。
     * 好在每个位置上的都是有'A', 'C', 'G', 'T'这四个碱基来的，所以我们可以用广度优先搜索，具体步骤如下：
     * 1 把begin放入队列中
     * 2 出队一个元素，修改这个元素上第一字母，修改值在这四个字母中选择'A', 'C', 'G', 'T'，四个字母都遍历一遍，如果和最后一个元素匹配，那么就退出，返回当前的层级（step）如果修改后元素的在bank的中出现，那么就放入队列中，同时删除bank中的相同的元素。
     * 3 然后把第一个元素还原原先的字母，然后开始修改第二个字母。执行和第2步一致。
     * 链接：https://leetcode-cn.com/problems/minimum-genetic-mutation/solution/javadan-xiang-yan-du-you-xian-sou-suo-he-shuang-xi/
     *
     * @param start 起始基因序列
     * @param end   目标基因序列
     * @param bank  基因库
     * @return
     */
    public int minMutation2(String start, String end, String[] bank) {
        HashSet<String> set = new HashSet<>(Arrays.asList(bank));
        if (!set.contains(end)) {
            return -1;
        }
        char[] four = {'A', 'C', 'G', 'T'};
        Queue<String> queue = new LinkedList<>();
        queue.offer(start);
        set.remove(start);
        int step = 0;
        while (queue.size() > 0) {
            System.out.println("*******bef当前的步数是:" + step + "*******");
            step++;
            System.out.println("*******aft当前的步数是:" + step + "*******");
            for (int count = queue.size(); count > 0; --count) {
                char[] temStringChars = queue.poll().toCharArray();
                for (int i = 0, len = temStringChars.length; i < len; ++i) {
                    char oldChar = temStringChars[i];
                    for (int j = 0; j < 4; ++j) {
                        temStringChars[i] = four[j];
                        String newGenetic = new String(temStringChars);
                        if (end.equals(newGenetic)) {
                            return step;
                        } else if (set.contains(newGenetic)) {
                            set.remove(newGenetic);
                            queue.offer(newGenetic);
                        }
                    }
                    temStringChars[i] = oldChar;
                }
            }
        }
        return -1;
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
     * 322. 零钱兑换
     * 给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
     * 你可以认为每种硬币的数量是无限的
     * <p>
     * 方法二：动态规划-自上而下 [通过]
     * https://leetcode-cn.com/problems/coin-change/solution/322-ling-qian-dui-huan-by-leetcode-solution/
     */
    public int coinChange4(int[] coins, int amount) {
        if (amount < 1) {
            return 0;
        }
        return coinChange(coins, amount, new int[amount]);
    }

    private int coinChange(int[] coins, int rem, int[] count) {
        if (rem < 0) {
            return -1;
        }
        if (rem == 0) {
            return 0;
        }
        if (count[rem - 1] != 0) {
            return count[rem - 1];
        }
        int min = Integer.MAX_VALUE;
        for (int coin : coins) {
            int res = coinChange(coins, rem - coin, count);
            if (res >= 0 && res < min) {
                min = 1 + res;
            }
        }
        count[rem - 1] = (min == Integer.MAX_VALUE) ? -1 : min;
        return count[rem - 1];
    }

    /**
     * 方法三：动态规划：自下而上 [通过]
     * https://leetcode-cn.com/problems/coin-change/solution/322-ling-qian-dui-huan-by-leetcode-solution/
     */
    public int coinChange5(int[] coins, int amount) {
        int max = amount + 1;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, max);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    /**
     * *Java* Both iterative and recursive solutions with explanations
     * *****************************************************************************************************************
     * #Recursive Method:#
     * The idea is very classic dynamic programming: think of the last step we take. Suppose we have already found out the best way to sum up to amount a, then for the last step, we can choose any coin type which gives us a remainder r where r = a-coins[i] for all i's. For every remainder, go through exactly the same process as before until either the remainder is 0 or less than 0 (meaning not a valid solution). With this idea, the only remaining detail is to store the minimum number of coins needed to sum up to r so that we don't need to recompute it over and over again.
     * *****************************************************************************************************************
     * Code in Java:
     */
    public int coinChange(int[] coins, int amount) {
        if (amount < 1) return 0;
        return helper(coins, amount, new int[amount]);
    }

    private int helper(int[] coins, int rem, int[] count) { // rem: remaining coins after the last step; count[rem]: minimum number of coins to sum up to rem
        if (rem < 0) return -1; // not valid
        if (rem == 0) return 0; // completed
        if (count[rem - 1] != 0) return count[rem - 1]; // already computed, so reuse
        int min = Integer.MAX_VALUE;
        for (int coin : coins) {
            int res = helper(coins, rem - coin, count);
            if (res >= 0 && res < min)
                min = 1 + res;
        }
        count[rem - 1] = (min == Integer.MAX_VALUE) ? -1 : min;
        return count[rem - 1];
    }

    /**
     * If you are interested in my other posts, please feel free to check my Github page here: https://github.com/F-L-A-G/Algorithms-in-Java
     * *****************************************************************************************************************
     * #Iterative Method:#
     * For the iterative solution, we think in bottom-up manner. Suppose we have already computed all the minimum counts up to sum, what would be the minimum count for sum+1?
     * *****************************************************************************************************************
     * Code in Java:
     */
    public int coinChange2(int[] coins, int amount) {
        if (amount < 1) return 0;
        int[] dp = new int[amount + 1];
        int sum = 0;

        while (++sum <= amount) {
            int min = -1;
            for (int coin : coins) {
                if (sum >= coin && dp[sum - coin] != -1) {
                    int temp = dp[sum - coin] + 1;
                    min = min < 0 ? temp : (temp < min ? temp : min);
                }
            }
            dp[sum] = min;
        }
        return dp[amount];
    }

    /**
     * For those who can't understand the code of Iterative method (Personally, I think it is hard to read)
     */
    public int coinChange3(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        for (int i = 1; i <= amount; i++) {
            int min = Integer.MAX_VALUE;
            for (int coin : coins) {
                if (i - coin >= 0 && dp[i - coin] != -1)
                    min = dp[i - coin] < min ? dp[i - coin] : min;
            }
            // Set dp[i] to -1 if i (current amount) can not be reach by  coins array
            dp[i] = min == Integer.MAX_VALUE ? -1 : 1 + min;
        }
        return dp[amount];
    }

    /**
     * 860. Lemonade Change
     * Intuition:
     * When the customer gives us $20, we have two options:
     * 1.To give three $5 in return
     * 2.To give one $5 and one $10.
     * On insight is that the second option (if possible) is always better than the first one.
     * Because two $5 in hand is always better than one $10
     * Explanation:
     * Count the number of $5 and $10 in hand.
     * <p>
     * if (customer pays with $5) five++;
     * if (customer pays with $10) ten++, five--;
     * if (customer pays with $20) ten--, five-- or five -= 3;
     * <p>
     * Check if five is positive, otherwise return false.
     * Time Complexity
     * Time O(N) for one iteration
     * Space O(1)
     */
    public boolean lemonadeChange(int[] bills) {
        int five = 0, ten = 0;
        for (int i : bills) {
            if (i == 5) five++;
            else if (i == 10) {
                five--;
                ten++;
            } else if (ten > 0) {
                ten--;
                five--;
            } else five -= 3;
            if (five < 0) return false;
        }
        return true;
    }

    /**
     * 860. 柠檬水找零
     * 输入：[5,5,5,10,20]
     * 输出：true
     */
    public boolean lemonadeChange2(int[] bills) {
        int five = 0, ten = 0;
        for (int bill : bills) {
            if (bill == 5) {
                five++;
            } else if (bill == 10) {
                if (five == 0) {
                    return false;
                }
                five--;
                ten++;
            } else {
                if (five > 0 && ten > 0) {
                    five--;
                    ten--;
                } else if (five >= 3) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }


    /**
     * 122. 买卖股票的最佳时机 II
     * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     * 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     *
     * 暴力搜索、贪心算法、动态规划（Java）
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/solution/tan-xin-suan-fa-by-liweiwei1419-2/
     * 推荐阅读 @stormsunshine 编写的文章《股票问题系列通解（转载翻译）》
     * 链接：https://leetcode-cn.com/circle/article/qiAgHn/
     *
     * 输入: [7,1,5,3,6,4]
     * 输出: 7
     */
    /**
     * 方法一：暴力搜索（超时）
     * 根据题意：由于不限制交易次数，在每一天，就可以根据当前是否持有股票选择相应的操作。「暴力搜索」在树形问题里也叫「回溯搜索」、「回溯法」。
     * 因为超时所以不采取以下这种方式
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/solution/tan-xin-suan-fa-by-liweiwei1419-2/
     */
    private int res;

    public int maxProfit(int[] prices) {
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
     * <p>
     * 根据 「力扣」第 121 题的思路，需要设置一个二维矩阵表示状态。
     * 第 1 步：定义状态
     * 状态 dp[i][j] 定义如下：
     * dp[i][j] 表示到下标为 i 的这一天，持股状态为 j 时，我们手上拥有的最大现金数。
     * 注意：限定持股状态为 j 是为了方便推导状态转移方程，这样的做法满足 无后效性。
     * 其中：
     * 第一维 i 表示下标为 i 的那一天（ 具有前缀性质，即考虑了之前天数的交易 ）；
     * 第二维 j 表示下标为 i 的那一天是持有股票，还是持有现金。这里 0 表示持有现金（cash），1 表示持有股票（stock）。
     * <p>
     * 第 2 步：思考状态转移方程
     * 状态从持有现金（cash）开始，到最后一天我们关心的状态依然是持有现金（cash）；
     * 每一天状态可以转移，也可以不动。状态转移用下图表示：
     * 如下链接所示
     * 链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/solution/tan-xin-suan-fa-by-liweiwei1419-2/
     * （状态转移方程写在代码中）
     * <p>
     * 说明：
     * 由于不限制交易次数，除了最后一天，每一天的状态可能不变化，也可能转移；
     * 写代码的时候，可以不用对最后一天单独处理，输出最后一天，状态为 0 的时候的值即可。
     * <p>
     * 第 3 步：确定初始值
     * 起始的时候：
     * 如果什么都不做，dp[0][0] = 0；
     * 如果持有股票，当前拥有的现金数是当天股价的相反数，即 dp[0][1] = -prices[i]；
     * <p>
     * 第 4 步：确定输出值
     * 终止的时候，上面也分析了，输出 dp[len - 1][0]，因为一定有 dp[len - 1][0] > dp[len - 1][1]。
     * <p>
     * 参考代码 2：
     * prices[i]代表第i天股价
     */
    public int maxProfit2(int[] prices) {
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
     */
    /**
     * 我们也可以将状态数组分开设置。
     * 参考代码 3：
     */
    public int maxProfit4(int[] prices) {
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

    public int maxProfit3(int[] prices) {
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
     * The greedy pair-wise approach mentioned in other posts is great for this problem indeed, but if we're not allowed to buy and sell stocks within the same day it can't be applied (logically, of course; the answer will be the same).
     * Actually, the straight-forward way of finding next local minimum and next local maximum is not much more complicated, so, just for the sake of having an alternative I share the code in Java for such case.
     */
    public int maxProfit7(int[] prices) {
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
    public int maxProfit8(int[] prices) {
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
     * 分发饼干（亚马逊在半年内面试中考过）
     * 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
     * 对每个孩子 i，都有一个胃口值g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j]。如果 s[j]>= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。
     * 你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
     *
     * 思路:
     * 为了了满足更多的小孩，就不要造成饼干尺寸的浪费。
     * 大尺寸的饼干既可以满足胃口大的孩子也可以满足胃口小的孩子，那么就应该优先满足胃口大的。
     * 这里的局部最优就是大饼干喂给胃口大的，充分利用饼干尺寸喂饱一个，全局最优就是喂饱尽可能多的小孩。
     * 可以尝试使用贪心策略，先将饼干数组和小孩数组排序。
     * 然后从后向前遍历小孩数组，用大饼干优先满足胃口大的，并统计满足小孩数量。
     * 链接：https://leetcode-cn.com/problems/assign-cookies/solution/455-fen-fa-bing-gan-tan-xin-jing-dian-ti-mu-xiang-/
     *
     * 解题思路
     * 对孩子数组与饼干数组从小到大排序
     * 初始化孩子下标为0
     * 遍历饼干数组，符合条件则孩子下标 +1
     */
    /**
     * 普通的双指针解法 !!!!!!  链接：https://leetcode-cn.com/problems/assign-cookies/solution/shuang-zhi-zhen-by-xing-yun-han-han/
     */
    public int findContentChildren(int[] g, int[] s) {
        int index_g = 0;
        int index_s = 0;
        //先对两个数组排序，g是胃口大小数组，s是饼干大小数组
        Arrays.sort(g);
        Arrays.sort(s);
        //这里循环的条件是两个指针不达到两个数组的长度，也就是没有元素移动了
        while (index_g < g.length && index_s < s.length) {
            if (g[index_g] <= s[index_s]) {
                //当胃口比饼干小，那么匹配成功，分给小孩
                index_g++;
            }
            index_s++;
        }
        return index_g;
    }

    /**
     * Here's a slightly more readable version:
     */
    public int findContentChildren2(int[] children, int[] cookies) {
        Arrays.sort(children);
        Arrays.sort(cookies);
        int child = 0;
        for (int cookie = 0; child < children.length && cookie < cookies.length; cookie++) {
            if (cookies[cookie] >= children[child]) {
                child++;
            }
        }
        return child;
    }

    /**
     * Simple Greedy Java Solution
     * Just assign the cookies starting from the child with less greediness to maximize the number of happy children .
     */
    public int findContentChildren3(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int i = 0;
        for (int j = 0; i < g.length && j < s.length; j++) {
            if (g[i] <= s[j]) i++;
        }
        return i;
    }

    /**
     * 874. 模拟行走机器人
     * 机器人在一个无限大小的网格上行走，从点(0, 0) 处开始出发，面向北方。该机器人可以接收以下三种类型的命令：
     * -2：向左转 90 度
     * -1：向右转 90 度
     * 1 <= x <= 9：向前移动x个单位长度
     * 在网格上有一些格子被视为障碍物。
     * 第 i 个障碍物位于网格点 (obstacles[i][0], obstacles[i][1])
     * 机器人无法走到障碍物上，它将会停留在障碍物的前一个网格方块上，但仍然可以继续该路线的其余部分。
     * 返回从原点到机器人所有经过的路径点（坐标为整数）的最大欧式距离的平方。
     * 示例 1：
     * 输入: commands = [4,-1,3], obstacles = []
     * 输出: 25
     * 解释: 机器人将会到达 (3, 4)
     * 示例 2：
     * 输入: commands = [4,-1,4,-2,4], obstacles = [[2,4]]
     * 输出: 65
     * 解释: 机器人在左转走到 (1, 8) 之前将被困在 (1, 4) 处
     * <p>
     * 输入：commands 和 obstacles，其中 obstacles = [[2,4]] 的意思是坐标点(2,4)代表障碍物的坐标
     * 输出：机器人所经过的每个坐标点(x,y)到原点的欧式距离的平方的最大值
     * 欧式距离: sqrt {x^2+y^2}
     * 欧式距离的平方: {x^2+y^2}
     * 图解如下:
     * https://leetcode-cn.com/problems/walking-robot-simulation/solution/tu-jie-mo-ni-xing-zou-ji-qi-ren-by-dekeshile/
     * 如上图所示：
     * 机器人初始位置为坐标点(0,0)，初始方向为向北
     * 读取第一个指令为4，沿着当前方向“北”，向前走4个单位，停在坐标点(0,4)
     * 读取第二个指令-1，该指令表示“向右转90度”，那么机器人就由原来的“北”右转90度之后方向变为“东”
     * 读取第三个指令4，沿着当前方向“东”，向前走4个单位，但是发现坐标点(2,4)是一个障碍物，不能跨越障碍物，
     * 只能停留在障碍物前面一个单位，即坐标点(1,4)
     * 读取第四个指令-2，该指令表示“向左转90度”，那么机器人就由原来的“东”左转90度之后方向变为“北”
     * 读取第五个指令4，沿着当前方向“北”，向前走4个单位，停在坐标点(1,8)
     * 65怎么得来的？ 机器人所经过的这些点中，坐标点(1,8)计算出的欧式距离的平方最大，为 1^2+8^2=65
     * 总体思想：模拟机器人行走过程，计算每一步坐标点到原点的欧式距离的平方，与保存的最大值比较，实时更新最大值
     * 具体的：
     * 1.分解机器人行走
     * 走k步，就是朝着一个方向走k个1步
     * 怎么朝着某个方向走出一步
     * 方向向北，机器人坐标点向上走一步
     * 方向向东，机器人坐标点向右走一步
     * 方向向南，机器人坐标点向下走一步
     * 方向向西，机器人坐标点向左走一步
     * int direx[] = {0,1,0,-1};
     * int direy[] = {1,0,-1,0};
     * direx[],direy[] 要竖着对齐看
     * - 向北，坐标轴上x不动，y+1, 即(0,1)
     * - 向东，坐标轴上x+1，y不动, 即(1,0)
     * - 向南，坐标轴上x不动，y-1, 即(0,-1)
     * - 向西，坐标轴上x-1，y不动, 即(-1,0)
     * 走( direx[i], direy[i] )，加上当前坐标后为 (curx,cury) + ( direx[i], direy[i] )
     * 2.机器人如何调整方向
     * direx[]direy[] 的下标 i 代表了当前机器人的方向
     * i=0,向北
     * i=1,向东
     * i=2,向南
     * i=3,向西
     * 当读取到调整方向的指令时，如
     * "-1"：“向右转90度”，只要当前方向curdire + 1就可以得到右转方向
     * "-2"：“向左转90度”，只要当前方向curdire + 3 就可以得到左转方向 (curdire + 3) % 4，
     * 因为不管curdire当前是哪个方向，左转都在其左边，在direx数组的定义中顺时针数3个就是其左边，所以就是加3
     * 3.怎么判断是否遇到了障碍物
     * 障碍物有多个，所以需要有一个障碍物坐标点集合
     * 机器人每试图走一个位置，就用此位置与障碍物集合列表里的坐标进行比较，看是否刚好是障碍物坐标点
     * 不是，则“真正走到这个点”，更新机器人坐标点(curx,cury)
     * 是障碍物，那么不走下一步，停留在当前，执行下一条命令
     * 注：
     * set 和 unordered_set 底层分别是用红黑树和哈希表实现的。
     * unordered_set 不能用来保存 pair<int, int>，但是 set 可以。
     * 因为 unordered_set 是基于哈希的，而 C++ 并没有给 pair 事先写好哈希方法。
     * set 是基于比较的树结构，所以 pair 里的数据结构只要都支持比较就能储存。
     * 链接：https://leetcode-cn.com/problems/walking-robot-simulation/solution/tu-jie-mo-ni-xing-zou-ji-qi-ren-by-dekeshile/
     * 备注：为什么要 %4 ？ 方向只有4个,我们建立的 direx 和 direy 只有4个元素  否找数组下标越界;  因为要确保curdire的值只能是 0,1,2,3 ;
     */
    public int robotSim(int[] commands, int[][] obstacles) {
        int[] direx = {0, 1, 0, -1};
        int[] direy = {1, 0, -1, 0};
        int curx = 0, cury = 0;
        int curdire = 0;
        int comLen = commands.length;
        int ans = 0;
        Set<Pair<Integer, Integer>> obstacleSet = new HashSet<Pair<Integer, Integer>>();
        // Replace with enhanced 'for' 把下列注释掉的代码简化成一行代码。即: for (int[] obstacle : obstacles) obstacleSet.add(new Pair<>(obstacle[0], obstacle[1]));
        // for (int i = 0; i < obstacles.length; i++) {
        //     obstacleSet.add(new Pair<>(obstacles[i][0], obstacles[i][1]));
        // }
        for (int[] obstacle : obstacles) obstacleSet.add(new Pair<>(obstacle[0], obstacle[1]));
        for (int command : commands) {
            if (command == -1) {  // -1：向右转 90 度
                curdire = (curdire + 1) % 4;
            } else if (command == -2) {  // -2：向左转 90 度
                curdire = (curdire + 3) % 4;
            } else {  // 1 <= x <= 9：向前移动 x 个单位长度
                for (int j = 0; j < command; j++) {
                    //试图走出一步，并判断是否遇到了障碍物，
                    int nx = curx + direx[curdire];
                    int ny = cury + direy[curdire];
                    //当前坐标不是障碍点，计算并与存储的最大欧式距离的平方做比较
                    if (!obstacleSet.contains(new Pair<Integer, Integer>(nx, ny))) {
                        curx = nx;
                        cury = ny;
                        ans = Math.max(ans, curx * curx + cury * cury);
                    } else {
                        //是障碍点，被挡住了，停留，只能等待下一个指令，那可以跳出当前指令了
                        break;
                    }
                }
            }
        }
        return ans;
    }

    /**
     * The robot starts at point (0, 0) and faces north. Which edge of grid is to the north?
     * Since it will go to point (3, 4) with commands = [4,-1,3], obstacles = [], we know that the right edge is to the North.
     * W
     * S -|- N
     * E
     * How do we represent absolute orientations given only relative turning directions(i.e., left or right)? We define direction indicates the absolute orientation as below:
     * North, direction = 0, directions[direction] = {0, 1}
     * East,  direction = 1, directions[direction] = {1, 0}
     * South, direction = 2, directions[direction] = {0, -1}
     * West,  direction = 3, directions[direction] = {-1, 0}
     * direction will increase by one when we turn right,
     * and will decrease by one (or increase by three) when we turn left.
     * https://leetcode.com/problems/walking-robot-simulation/discuss/155682/Logical-Thinking-with-Clear-Code
     */
    public int robotSim2(int[] commands, int[][] obstacles) {
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        // Set of obstacles indexes in the format of : obstacle[0] + " " + obstacle[1]
        Set<String> obstaclesSet = new HashSet<>();
        for (int[] obstacle : obstacles) {
            obstaclesSet.add(obstacle[0] + " " + obstacle[1]);
        }

        int x = 0, y = 0, direction = 0, maxDistSquare = 0;
        for (int i = 0; i < commands.length; i++) {
            if (commands[i] == -2) { // Turns left
                direction = (direction + 3) % 4;
            } else if (commands[i] == -1) { // Turns right
                direction = (direction + 1) % 4;
            } else { // Moves forward commands[i] steps
                int step = 0;
                while (step < commands[i] && (!obstaclesSet.contains((x + directions[direction][0]) + " " + (y + directions[direction][1])))) {
                    x += directions[direction][0];
                    y += directions[direction][1];
                    step++;
                }
            }
            maxDistSquare = Math.max(maxDistSquare, x * x + y * y);
        }

        return maxDistSquare;
    }

    /**
     * 55. 跳跃游戏
     * 给定一个非负整数数组，你最初位于数组的第一个位置。
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * 判断你是否能够到达最后一个位置。
     * 示例1:
     * 输入: [2,3,1,1,4]
     * 输出: true
     * 解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
     * 示例2:
     * 输入: [3,2,1,0,4]
     * 输出: false
     * 解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
     */
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int rightmost = 0;
        for (int i = 0; i < n; ++i) {
            if (i <= rightmost) {
                rightmost = Math.max(rightmost, i + nums[i]);
                if (rightmost >= n - 1) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Python击败97%，看不懂你锤我
     * def canJump(self, nums) :
     *     max_i = 0       #初始化当前能到达最远的位置
     *     for i, jump in enumerate(nums):   #i为当前位置，jump是当前位置的跳数
     *         if max_i>=i and i+jump>max_i:  #如果当前位置能到达，并且当前位置+跳数>最远位置
     *            max_i = i+jump  #更新最远能到达位置
     *
     *     return max_i>=i
     * 思路：尽可能到达最远位置（贪心）。
     * 如果能到达某个位置，那一定能到达它前面的所有位置。
     * #初始化当前能到达最远的位置
     * #i为当前位置，jump是当前位置的跳数
     * #如果当前位置能到达，并且当前位置+跳数>最远位置
     * #更新最远能到达位置
     * 方法：初始化最远位置为 0，然后遍历数组，如果当前位置能到达，并且当前位置+跳数>最远位置，就更新最远位置。最后比较最远位置和数组长度。
     * 复杂度：时间复杂度 O(n)，空间复杂度 O(1)。
     * 链接：https://leetcode-cn.com/problems/jump-game/solution/pythonji-bai-97kan-bu-dong-ni-chui-wo-by-mo-lan-4/
     */
    /**
     * https://leetcode.com/problems/jump-game/discuss/20907/1-6-lines-O(n)-time-O(1)-space
     * 1-6 lines, O(n) time, O(1) space
     * Solution 1
     * Going forwards. m tells the maximum index we can reach so far.
     * def canJump(self, nums):
     * m = 0
     * for i, n in enumerate(nums):
     * if i > m:
     * return False
     * m = max(m, i+n)
     * return True
     * Solution 2
     * One-liner version:
     * def canJump(self, nums):
     * return reduce(lambda m, (i, n): max(m, i+n) * (i <= m), enumerate(nums, 1), 1) > 0
     */
    public boolean canJump2(int[] nums) {
        int n = nums.length, farest = 0;
        for (int i = 0; i < n; i++) {
            if (farest < i) return false;
            farest = Math.max(i + nums[i], farest);
        }

        return true;
    }

    /**
     * 超哥:从后往前
     */
    public boolean canJump3(int[] nums) {
        if (nums == null) {
            return false;
        }
        int canReachable = nums.length - 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            if (nums[i] + i >= canReachable) {
                canReachable = i;
            }
        }
        return canReachable == 0;
    }

    /**
     * Linear and simple solution in C++
     * i <= reach, the furthest "start point" that I can reach
     * start point + nums[i] >= n, win!
     * start point + nums[i] < n, lose!
     * so greedy
     */
    public boolean canJump4(int[] nums) {
        int i = 0, n = nums.length;
        for (int reach = 0; i < n && i <= reach; ++i)
            reach = Math.max(i + nums[i], reach);
        return i == n;
    }

    /**
     * 45. 跳跃游戏 II
     * 给定一个非负整数数组，你最初位于数组的第一个位置。
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * 你的目标是使用最少的跳跃次数到达数组的最后一个位置。
     * 输入: [2,3,1,1,4]
     * 输出: 2
     * 解释: 跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
     * ***********************************************************************************************
     * 1.DP
     * 解题思路：
     * 利用一个数组，表示到达每个位置最少跳跃次数。
     * 遍历数组，然后对每个位置可以跳到的范围内的位置进行状态转移，保留次数少的。
     * 时间复杂度是n方的。
     * 2.贪心
     * 解题思路：维护一个范围内的最少跳跃次数，当超出该范围，那就不得不增加跳跃次数了。
     */
    public int jump(int[] nums) {
        int n = nums.length;
        int[] f = new int[n];
        Arrays.fill(f, Integer.MAX_VALUE / 2);
        f[0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j <= nums[i] + i && j < n; j++) {
                // 状态转移，比较从前一个位置跳过来的次数
                f[j] = Math.min(f[i] + 1, f[j]);
            }
        }
        return f[n - 1];
    }

    public int jump2(int[] nums) {
        int res = 0;
        int max = 0, end = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            max = Math.max(max, nums[i] + i);
            // 到达一个范围的终点，说明之前一跳不可能超过这个位置，所以跳跃次数必须增加
            if (end == i) {
                res += 1;
                end = max;
            }
        }
        return res;
    }

    /**
     * 求最高得分
     * 给你一个下标从 0 开始的整数数组 nums 和一个整数 k 。
     * 一开始你在下标 0 处。每一步，你最多可以往前跳 k 步，但你不能跳出数组的边界。也就是说，你可以从下标 i 跳到 [i + 1， min(n - 1, i + k)] 包含 两个端点的任意位置。
     * 你的目标是到达数组最后一个位置（下标为 n - 1 ），你的 得分 为经过的所有数字之和。
     * 请你返回你能得到的 最大得分 。
     */
    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        Deque<Integer> q = new LinkedList<>();
        int[] f = new int[n];
        // 填入初始值
        f[0] = nums[0];
        q.offerLast(f[0]);
        for (int i = 1; i < n; i++) {
            // 超出步数的元素出队
            if (i > k && q.peekFirst() == f[i - k - 1]) q.poll();
            // 计算当前值
            f[i] = nums[i] + q.peekFirst();
            // 当前值入队
            while (!q.isEmpty() && q.peekLast() <= f[i]) q.pollLast();
            q.offerLast(f[i]);

        }
        return f[n - 1];
    }

    /**
     * 69. x 的平方根（字节跳动、微软、亚马逊在半年内面试中考过）
     * 实现 int sqrt(int x) 函数。
     * 计算并返回 x 的平方根，其中 x 是非负整数。
     * 由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
     * *****************************************************************************************************************
     * 3 JAVA solutions with explanation
     * The three solutions are as the follows, solution1 and solution3 are pretty straight forward.
     * Look for the critical point: i * i <= x && (i+1)(i+1) > x
     * A little trick is using i <= x / i for comparison, instead of i * i <= x, to avoid exceeding integer upper limit.
     * *****************************************************************************************************************
     * Solution1 - Binary Search Solution: Time complexity = O(lg(x)) = O(32)=O(1)
     */
    public int mySqrt(int x) {
        if (x == 0) return 0;
        int start = 1, end = x;
        while (start < end) {
            int mid = start + (end - start) / 2;
            if (mid <= x / mid && (mid + 1) > x / (mid + 1))// Found the result
                return mid;
            else if (mid > x / mid)// Keep checking the left part
                end = mid;
            else
                start = mid + 1;// Keep checking the right part
        }
        return start;
    }

    /**
     * Solution2 - Newton Solution: Time complexity = O(lg(x))
     * I think Newton solution will not be faster than Solution1(Binary Search), because i = (i + x / i) / 2, the two factors i and x / i are with opposite trends. So time complexity in the best case is O(lgx).
     * Anyone can give the accurate time complexity? Appreciate it!
     */
    public int mySqrt2(int x) {
        if (x == 0) return 0;
        long i = x;
        while (i > x / i)
            i = (i + x / i) / 2;
        return (int) i;
    }

    /**
     * Solution3 - Brute Force: Time complexity = O(sqrt(x))
     */
    public int mySqrt3(int x) {
        if (x == 0) return 0;
        for (int i = 1; i <= x / i; i++)
            if (i <= x / i && (i + 1) > x / (i + 1))// Look for the critical point: i*i <= x && (i+1)(i+1) > x
                return i;
        return -1;
    }

    /**
     * *******************************************************************************************************************************************************
     * 69. Sqrt(x)
     * Given a non-negative integer x, compute and return the square root of x.
     * Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.
     * *******************************************************************************************************************************************************
     * Quite a few people used Newton already, but I didn't see someone make it this short. Same solution in every language. Explanation under the solutions.
     *
     * C++ and C
     *
     *     long r = x;
     *     while (r*r > x)
     *         r = (r + x/r) / 2;
     *     return r;
     * Python
     *
     *     r = x
     *     while r*r > x:
     *         r = (r + x/r) / 2
     *     return r
     * Ruby
     *
     *     r = x
     *     r = (r + x/r) / 2 while r*r > x
     *     r
     * Java and C#
     *
     *     long r = x;
     *     while (r*r > x)
     *         r = (r + x/r) / 2;
     *     return (int) r;
     * JavaScript
     *
     *     r = x;
     *     while (r*r > x)
     *         r = ((r + x/r) / 2) | 0;
     *     return r;
     * *******************************************************************************************************************************************************
     * Explanation
     * Apparently, using only integer division for the Newton method works. And I guessed that if I start at x, the root candidate will decrease monotonically and never get too small.
     * The above solutions all got accepted, and in C++ I also verified it locally on my PC for all possible inputs (0 to 2147483647):
     * Newton 牛顿迭代法 By StefanPochmann 光头哥
     */
    public int mySqrt4(int x) {
        long r = x;
        while (r * r > x)
            r = (r + x / r) / 2;
        return (int) r;
    }

    /**
     * 超哥的解法:二分查找
     * a.单调
     * b.边界
     * c.index
     */
    public int mySqrt5(int x) {
        if (x == 0 || x == 1) return x;
        long left = 1, right = x, mid = 1;
        while (left <= right) {
            mid = left + (right - left) / 2;
            if (mid * mid > x) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return (int) right;
    }

    /**
     * 367. 有效的完全平方数（亚马逊在半年内面试中考过）
     * 给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。
     * 说明：不要使用任何内置的库函数，如 sqrt。
     * 方法二：牛顿迭代法[阅读一下]
     * 牛顿迭代法：公式是如何推导的呢？让我们做一个非常粗略的推导。
     * 问题是找出：f(x) = x^2 - num = 0 的根。
     * 牛顿迭代法的思想是从一个初始近似值开始，然后作一系列改进的逼近根的过程。
     * https://leetcode-cn.com/problems/valid-perfect-square/solution/you-xiao-de-wan-quan-ping-fang-shu-by-leetcode/
     * 第四招：牛顿迭代法[阅读一下]
     * https://leetcode-cn.com/problems/valid-perfect-square/solution/ceng-ceng-di-jin-zhu-bu-zui-you-de-si-chong-jie-fa/
     */
    public boolean isPerfectSquare(int num) {
        if (num < 2) return true;

        long x = num / 2;
        while (x * x > num) {
            x = (x + num / x) / 2;
        }
        return (x * x == num);
    }

    /**
     * This is a math problem：
     * 1 = 1
     * 4 = 1 + 3
     * 9 = 1 + 3 + 5
     * 16 = 1 + 3 + 5 + 7
     * 25 = 1 + 3 + 5 + 7 + 9
     * 36 = 1 + 3 + 5 + 7 + 9 + 11
     * ....
     * so 1+3+...+(2n-1) = (2n-1 + 1)n/2 = n^2
     * *****************************************************************************************************************
     * a1 = 1; a2 = 3; a3 = 5... an = 2n -1;
     * Sn = (a1 + an) * n / 2 = (1 + 2n - 1) * n / 2 = n^2
     * *****************************************************************************************************************
     * (a+b)/2 >= sqrt(a*b) only when a = b the left is equal to the right
     */
    public boolean isPerfectSquare2(int num) {
        int i = 1;
        while (num > 0) {
            num -= i;
            i += 2;
        }
        return num == 0;
    }

    /**
     * Can someone explain me why I had a similar version, but just do addition, however, I got TLE:
     * public boolean isPerfectSquare(int num) {
     *     int i = 1, temp = 1;
     *     while(temp < num){
     *         i += 2;
     *         temp += i;
     *     }
     *     return temp == num;
     * }
     * diff of long and int. in Int it will cause overflow.
     * Because it will cause the overflow problem. The maximum of a integer in java is 2147483647.
     *
     * Version: Don't use long type
     * int res = num / mid, remain = num % mid;  this is awsm
     * [Java] Binary Search - Clean code - logN
     */
    public boolean isPerfectSquare3(int num) {
        int left = 1, right = num;
        while (left <= right) {
            int mid = left + (right - left) / 2; // to avoid overflow incase (left+right)>2147483647
            int res = num / mid, remain = num % mid;
            if (res == mid && remain == 0) return true; // check if mid * mid == num
            if (res > mid) { // mid is small -> go right to increase mid
                left = mid + 1;
            } else {
                right = mid - 1; // mid is large -> to left to decrease mid
            }
        }
        return false;
    }
    /**
     * 515. Find Largest Value in Each Tree Row
     * Given the root of a binary tree, return an array of the largest value in each row of the tree (0-indexed).
     * <p>
     * Just a simple pre-order traverse idea. Use depth to expand result list size and put the max value in the appropriate position.
     */
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        helper(root, res, 0);
        return res;
    }

    private void helper(TreeNode root, List<Integer> res, int d) {
        if (root == null) {
            return;
        }
        //expand list size
        if (d == res.size()) {
            res.add(root.val);
        } else {
            //or set value
            res.set(d, Math.max(res.get(d), root.val));
        }
        helper(root.left, res, d + 1);
        helper(root.right, res, d + 1);
    }

    /**
     * BFS
     */
    public List<Integer> largestValues2(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        List<Integer> values = new ArrayList<Integer>();

        if (root != null) q.offer(root);

        while (!q.isEmpty()) {
            int max = Integer.MIN_VALUE, n = q.size();
            for (int i = 0; i < n; i++) {
                TreeNode node = q.poll();
                max = Math.max(max, node.val);
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
            values.add(max);
        }

        return values;
    }

    /**
     * DFS
     * One reason DFS is better than BFS is that in BFS you're always maintaining a queue that could potentially grow up to 2^logN (leaves of a full and balanced binary tree) whereas in DFS the only cost you pay is the stack-space logN:
     */
    public List<Integer> largestValues3(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        dfs(res, root, 0);
        return res;
    }

    private void dfs(List<Integer> res, TreeNode root, int row) {
        if (root == null) return;
        setMax(res, root, row);
        dfs(res, root.left, row + 1);
        dfs(res, root.right, row + 1);
    }

    private void setMax(List<Integer> res, TreeNode root, int row) {
        if (row >= res.size()) res.add(root.val);
        else res.set(row, Math.max(root.val, res.get(row)));
    }

    /**
     * 515. 在每个树行中找最大值
     * java代码BFS和DFS两种解决思路以及图文分析
     * <p>
     * 一，首先是BFS，这个比较简单，就是一行一行的遍历，像下面图中这样，在每一行中找到最大值即可
     * https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row/solution/javadai-ma-bfshe-dfsliang-chong-jie-jue-si-lu-yi-j/
     */
    public List<Integer> largestValues4(TreeNode root) {
        //LinkedList实现队列
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> values = new ArrayList<>();
        if (root != null)
            queue.add(root);//入队
        while (!queue.isEmpty()) {
            int max = Integer.MIN_VALUE;
            int levelSize = queue.size();//每一层的数量
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();//出队
                max = Math.max(max, node.val);//记录每层的最大值
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
            }
            values.add(max);
        }
        return values;
    }

    /**
     * 127. 单词接龙
     * Java 双向 BFS
     * 链接：https://leetcode-cn.com/problems/word-ladder/solution/suan-fa-shi-xian-he-you-hua-javashuang-xiang-bfs23/
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) {
            return 0;
        }
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.offer(beginWord);
        visited.add(beginWord);
        int count = 0;
        while (queue.size() > 0) {
            int size = queue.size();
            ++count;
            for (int i = 0; i < size; ++i) {
                String start = queue.poll();
                for (String s : wordList) {
                    // 已经遍历的不再重复遍历
                    if (visited.contains(s)) {
                        continue;
                    }
                    // 不能转换的直接跳过
                    if (!canConvert(start, s)) {
                        continue;
                    }
                    // 用于调试
                    // System.out.println(count + ": " + start + "->" + s);
                    // 可以转换，并且能转换成 endWord，则返回 count
                    if (s.equals(endWord)) {
                        return count + 1;
                    }
                    // 保存访问过的单词，同时把单词放进队列，用于下一层的访问
                    visited.add(s);
                    queue.offer(s);
                }
            }
        }
        return 0;
    }

    public boolean canConvert(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int count = 0;
        for (int i = 0; i < s1.length(); ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                ++count;
                if (count > 1) {
                    return false;
                }
            }
        }
        return count == 1;
    }

    /**
     * 126. 单词接龙 II
     * Very fast codes. Beat 100%
     * https://leetcode.com/problems/word-ladder-ii/discuss/40475/My-concise-JAVA-solution-based-on-BFS-and-DFS
     */
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        Set<String> dict = new HashSet<>(wordList);
        List<List<String>> res = new ArrayList<>();
        if (!dict.contains(endWord)) {
            return res;
        }
        Map<String, List<String>> map = getChildren(beginWord, endWord, dict);
        List<String> path = new ArrayList<>();
        path.add(beginWord);
        findLadders(beginWord, endWord, map, res, path);
        return res;

    }

    public void findLadders(String beginWord, String endWord, Map<String, List<String>> map, List<List<String>> res, List<String> path) {
        if (beginWord.equals(endWord)) {
            res.add(new ArrayList<>(path));
        }
        if (!map.containsKey(beginWord)) {
            return;
        }
        for (String next : map.get(beginWord)) {
            path.add(next);
            findLadders(next, endWord, map, res, path);
            path.remove(path.size() - 1);
        }
    }

    public Map<String, List<String>> getChildren(String beginWord, String endWord, Set<String> dict) {
        Map<String, List<String>> map = new HashMap<>();
        Set<String> start = new HashSet<>();
        start.add(beginWord);
        Set<String> end = new HashSet<>();
        Set<String> visited = new HashSet<>();
        end.add(endWord);
        boolean found = false;
        boolean isBackward = false;
        while (!start.isEmpty() && !found) {
            if (start.size() > end.size()) {
                Set<String> tem = start;
                start = end;
                end = tem;
                isBackward = !isBackward;
            }
            Set<String> set = new HashSet<>();
            for (String cur : start) {
                visited.add(cur);
                for (String next : getNext(cur, dict)) {
                    if (visited.contains(next) || start.contains(next)) {
                        continue;
                    }
                    if (end.contains(next)) {
                        found = true;
                    }
                    set.add(next);
                    String parent = isBackward ? next : cur;
                    String child = isBackward ? cur : next;
                    if (!map.containsKey(parent)) {
                        map.put(parent, new ArrayList<>());
                    }
                    map.get(parent).add(child);

                }
            }
            start = set;
        }
        return map;

    }

    private List<String> getNext(String cur, Set<String> dict) {
        List<String> res = new ArrayList<>();
        char[] chars = cur.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            char old = chars[i];
            for (char c = 'a'; c <= 'z'; c++) {
                if (c == old) {
                    continue;
                }
                chars[i] = c;
                String next = new String(chars);
                if (dict.contains(next)) {
                    res.add(next);
                }
            }
            chars[i] = old;
        }
        return res;
    }

    /**
     * 超哥的二分查找代码模板
     * 二分查找条件:
     * a.单调
     * b.边界
     * c.index
     */
    public int binarySearch(int[] array, int target) {
        int left = 0, right = array.length - 1, mid;
        while (left <= right) {
            mid = (right - left) / 2 + left;

            if (array[mid] == target) {
                return mid;
            } else if (array[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }

    /**
     * 33. 搜索旋转排序数组
     * 升序排列的整数数组 nums 在预先未知的某个点上进行了旋转（例如， [0,1,2,4,5,6,7] 经旋转后可能变为 [4,5,6,7,0,1,2] ）。
     * 请你在数组中搜索 target ，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
     * Key idea:
     * if nums[mid] and target are on the same side -> easy life. If not, let's move towards the target side.
     * Also, (nums[mid]-nums[nums.length-1])*(target-nums[nums.length-1])>0 is more readable in my opinion
     */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) return -1;
        int lo = 0;
        int hi = nums.length - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            //target and mid are on the same side
            if ((nums[mid] - nums[nums.length - 1]) * (target - nums[nums.length - 1]) > 0) {
                if (nums[mid] < target)
                    lo = mid + 1;
                else
                    hi = mid;
            } else if (target > nums[nums.length - 1])
                hi = mid; // target on the left side
            else
                lo = mid + 1; // target on the right side
        }
        // now hi == lo
        if (nums[lo] == target)
            return lo;
        else
            return -1;
    }

    /**
     * 极简 Solution[阅读一下]
     * 二分查找
     * 以二分搜索为基本思路
     * 简要来说：
     * nums[0] <= nums[mid]（0 - mid不包含旋转）且nums[0] <= target <= nums[mid] 时 high 向前规约；
     * nums[mid] < nums[0]（0 - mid包含旋转），target <= nums[mid] < nums[0] 时向前规约（target 在旋转位置到 mid 之间）
     * nums[mid] < nums[0]，nums[mid] < nums[0] <= target 时向前规约（target 在 0 到旋转位置之间）
     * 其他情况向后规约
     * 也就是说nums[mid] < nums[0]，nums[0] > target，target > nums[mid] 三项均为真或者只有一项为真时向后规约。
     * 原文的分析是：
     * 注意到原数组为有限制的有序数组（除了在某个点会突然下降外均为升序数组）
     *
     * if nums[0] <= nums[I] 那么 nums[0] 到 nums[i] 为有序数组,那么当 nums[0] <= target <= nums[i] 时我们应该在 0-i0−i 范围内查找；
     * if nums[i] < nums[0] 那么在 0-i0−i 区间的某个点处发生了下降（旋转），那么 I+1I+1 到最后一个数字的区间为有序数组，并且所有的数字都是小于 nums[0] 且大于 nums[i]，当target不属于 nums[0] 到 nums[i] 时（target <= nums[i] < nums[0] or nums[i] < nums[0] <= target），我们应该在 0-i0−i 区间内查找。
     * 上述三种情况可以总结如下：
     * nums[0] <= target <= nums[i]
     *            target <= nums[i] < nums[0]
     *                      nums[i] < nums[0] <= target
     * 所以我们进行三项判断：
     * (nums[0] <= target)， (target <= nums[i]) ，(nums[i] < nums[0])，现在我们想知道这三项中有哪两项为真（明显这三项不可能均为真或均为假（因为这三项可能已经包含了所有情况））
     * 所以我们现在只需要区别出这三项中有两项为真还是只有一项为真。
     * 使用 “异或” 操作可以轻松的得到上述结果（两项为真时异或结果为假，一项为真时异或结果为真，可以画真值表进行验证）
     * 之后我们通过二分查找不断做小 target 可能位于的区间直到 low==high，此时如果 nums[low]==target 则找到了，如果不等则说明该数组里没有此项。
     *
     * 链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array/solution/ji-jian-solution-by-lukelee/
     * 超哥的解法:二分查找
     * a.单调
     * b.边界
     * c.index
     */
    public int search2(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if ((nums[0] > target) ^ (nums[0] > nums[mid]) ^ (target > nums[mid]))
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo == hi && nums[lo] == target ? lo : -1;
    }

    /**
     * [0,1,2,4,5,6,7]-->[4,5,6,7,0,1,2]
     * https://leetcode-cn.com/problems/search-in-rotated-sorted-array/solution/er-fen-fa-python-dai-ma-java-dai-ma-by-liweiwei141/
     */
    public int search3(int[] nums, int target) {
        int len = nums.length;
        if (len == 0) {
            return -1;
        }
        int left = 0;
        int right = len - 1;
        while (left < right) {
            int mid = left + (right - left + 1) / 2;
            if (nums[mid] < nums[right]) {
                // 使用上取整的中间数，必须在上面的 mid 表达式的括号里 + 1
                if (nums[mid] <= target && target <= nums[right]) {
                    // 下一轮搜索区间是 [mid, right]
                    left = mid;
                } else {
                    // 只要上面对了，这个区间是上面区间的反面区间，下一轮搜索区间是 [left, mid - 1]
                    right = mid - 1;
                }
            } else {
                // [left, mid] 有序，但是为了和上一个 if 有同样的收缩行为，
                // 我们故意只认为 [left, mid - 1] 有序
                // 当区间只有 2 个元素的时候 int mid = (left + right + 1) >>> 1; 一定会取到右边
                // 此时 mid - 1 不会越界，就是这么刚刚好
                if (nums[left] <= target && target <= nums[mid - 1]) {
                    // 下一轮搜索区间是 [left, mid - 1]
                    right = mid - 1;
                } else {
                    // 下一轮搜索区间是 [mid, right]
                    left = mid;
                }
            }
        }

        // 有可能区间内不存在目标元素，因此还需做一次判断
        if (nums[left] == target) {
            return left;
        }
        return -1;
    }

    /**
     * 74. 搜索二维矩阵（亚马逊、微软、Facebook 在半年内面试中考过）
     * 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
     * 每行中的整数从左到右按升序排列。
     * 每行的第一个整数大于前一行的最后一个整数。
     * 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 3
     * 输出：true
     * 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 13
     * 输出：false
     * 输入：matrix = [], target = 0
     * 输出：false
     * 链接：https://leetcode-cn.com/problems/search-a-2d-matrix
     * *****************************************************************************************************************
     * Don't treat it as a 2D matrix, just treat it as a sorted list
     * Use binary search.
     * n * m matrix convert to an array => matrix[x][y] => a[x * m + y]
     * an array convert to n * m matrix => a[x] =>matrix[x / m][x % m];
     * https://leetcode.com/problems/search-a-2d-matrix/discuss/26220/Don't-treat-it-as-a-2D-matrix-just-treat-it-as-a-sorted-list
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int n = matrix.length;
        int m = matrix[0].length;
        int l = 0, r = m * n - 1;
        while (l != r) {
            int mid = (l + r - 1) >> 1;
            if (matrix[mid / m][mid % m] < target)
                l = mid + 1;
            else
                r = mid;
        }
        return matrix[r / m][r % m] == target;
    }

    /**
     * Concise and straight forward! Cannot agree more.
     * Below is my Java solution:
     */
    public boolean searchMatrix2(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int start = 0, rows = matrix.length, cols = matrix[0].length;
        int end = rows * cols - 1;
        while (start <= end) {
            int mid = (start + end) / 2;
            if (matrix[mid / cols][mid % cols] == target) {
                return true;
            }
            if (matrix[mid / cols][mid % cols] < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return false;
    }

    /**
     * 153. 寻找旋转排序数组中的最小值（亚马逊、微软、字节跳动在半年内面试中考过）
     * 假设按照升序排序的数组在预先未知的某个点上进行了旋转。例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] 。
     * 请找出其中最小的元素。
     * 输入：nums = [3,4,5,1,2]
     * 输出：1
     */
    public int findMin(int[] nums) {
        if (nums == null) return -1;
        if (nums.length == 1) return nums[0];
        int p1 = 0, p2 = nums.length - 1;
        int mid = p1; // 假如旋转了数组的前面0个元素（也就是没有旋转），我们直接返回numbers[p1]
        while (nums[p1] > nums[p2]) {
            if (p2 - p1 == 1) {
                // 循环终止条件：当p2-p1=1时，p2所指元素为最小值
                mid = p2;
                break;
            }
            mid = (p1 + p2) / 2;
            if (nums[mid] > nums[p1]) p1 = mid;
            else p2 = mid;
        }
        return nums[mid];
    }

    /**
     * 一文解决 4 道「搜索旋转排序数组」题！
     * *****************************************************************************************************************
     * 本文涉及 4 道「搜索旋转排序数组」题：
     * LeetCode 33 题：搜索旋转排序数组
     * LeetCode 81 题：搜索旋转排序数组-ii
     * LeetCode 153 题：寻找旋转排序数组中的最小值
     * LeetCode 154 题：寻找旋转排序数组中的最小值-ii
     * 可以分为 3 类：
     * 33、81 题：搜索特定值
     * 153、154 题：搜索最小值
     * 81、154 题：包含重复元素
     * 链接：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/solution/yi-wen-jie-jue-4-dao-sou-suo-xuan-zhuan-pai-xu-s-3/
     * *****************************************************************************************************************
     * Java solution with binary search
     * The minimum element must satisfy one of two conditions: 1) If rotate, A[min] < A[min - 1]; 2) If not, A[0].
     * Therefore, we can use binary search: check the middle element, if it is less than previous one, then it is minimum.
     * If not, there are 2 conditions as well: If it is greater than both left and right element, then minimum element should be on its right, otherwise on its left.
     */
    public int findMin2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int start = 0, end = nums.length - 1;
        while (start < end) {
            int mid = (start + end) / 2;
            if (mid > 0 && nums[mid] < nums[mid - 1]) {
                return nums[mid];
            }
            if (nums[start] <= nums[mid] && nums[mid] > nums[end]) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return nums[start];
    }
}
