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
     *
     * Suppose we need to investigate about a mutation (mutation from "start" to "end"), where ONE mutation is defined as ONE single character changed in the gene string.
     *
     * For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation.
     *
     * Also, there is a given gene "bank", which records all the valid gene mutations. A gene must be in the bank to make it a valid gene string.
     *
     * Now, given 3 things - start, end, bank, your task is to determine what is the minimum number of mutations needed to mutate from "start" to "end". If there is no such a mutation, return -1.
     *
     * Note:
     *
     * Starting point is assumed to be valid, so it might not be included in the bank.
     * If multiple mutations are needed, all mutations during in the sequence must be valid.
     * You may assume start and end string is not the same.
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
            step++;
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
     * 860. Lemonade Change
     * Intuition:
     * When the customer gives us $20, we have two options:
     * 1.To give three $5 in return
     * 2.To give one $5 and one $10.
     * On insight is that the second option (if possible) is always better than the first one.
     * Because two $5 in hand is always better than one $10
     * Explanation:
     * Count the number of $5 and $10 in hand.
     *
     * if (customer pays with $5) five++;
     * if (customer pays with $10) ten++, five--;
     * if (customer pays with $20) ten--, five-- or five -= 3;
     *
     * Check if five is positive, otherwise return false.
     * Time Complexity
     * Time O(N) for one iteration
     * Space O(1)
     */
    public boolean lemonadeChange(int[] bills) {
        int five = 0, ten = 0;
        for (int i : bills) {
            if (i == 5) five++;
            else if (i == 10) {five--; ten++;}
            else if (ten > 0) {ten--; five--;}
            else five -= 3;
            if (five < 0) return false;
        }
        return true;
    }

    /**
     * The greedy pair-wise approach mentioned in other posts is great for this problem indeed, but if we're not allowed to buy and sell stocks within the same day it can't be applied (logically, of course; the answer will be the same).
     * Actually, the straight-forward way of finding next local minimum and next local maximum is not much more complicated, so, just for the sake of having an alternative I share the code in Java for such case.
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int profit = 0, i = 0;
        while (i < prices.length) {
            // find next local minimum
            while (i < prices.length-1 && prices[i+1] <= prices[i]) i++;
            int min = prices[i++]; // need increment to avoid infinite loop for "[1]"
            // find next local maximum
            while (i < prices.length-1 && prices[i+1] >= prices[i]) i++;
            profit += i < prices.length ? prices[i++] - min : 0;
        }
        return profit;
    }

    /**
     * The profit is the sum of sub-profits. Each sub-profit is the difference between selling at day j, and buying at day i (with j > i). The range [i, j] should be chosen so that the sub-profit is maximum:
     *
     * sub-profit = prices[j] - prices[i]
     *
     * We should choose j that prices[j] is as big as possible, and choose i that prices[i] is as small as possible (of course in their local range).
     *
     * Let's say, we have a range [3, 2, 5], we will choose (2,5) instead of (3,5), because 2<3.
     * Now, if we add 8 into this range: [3, 2, 5, 8], we will choose (2, 8) instead of (2,5) because 8>5.
     *
     * From this observation, from day X, the buying day will be the last continuous day that the price is smallest. Then, the selling day will be the last continuous day that the price is biggest.
     *
     * Take another range [3, 2, 5, 8, 1, 9], though 1 is the smallest, but 2 is chosen, because 2 is the smallest in a consecutive decreasing prices starting from 3.
     * Similarly, 9 is the biggest, but 8 is chosen, because 8 is the biggest in a consecutive increasing prices starting from 2 (the buying price).
     * Actually, the range [3, 2, 5, 8, 1, 9] will be splitted into 2 sub-ranges [3, 2, 5, 8] and [1, 9].
     *
     * We come up with the following code:
     * https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/discuss/208241/Explanation-for-the-dummy-like-me.
     */
    public int maxProfit2(int[] prices) {
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
}
