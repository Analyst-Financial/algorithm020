package algorithm;

import java.util.*;

public class Main {

    public static void main(String[] args) {

        System.out.println("--------------week4--------------");
        Week4 week4 = new Week4();
        List<String> wordList = new ArrayList<>();
        wordList.add("hot");
        wordList.add("dot");
        wordList.add("dog");
        wordList.add("lot");
        wordList.add("log");
        wordList.add("cog");

        String beginWord = "hit";
        String endWord = "cog";
        List<List<String>> res3 = week4.findLadders(beginWord, endWord, wordList);
        System.out.println(res3);

        /**
         * robotSim
         * North, direction = 0, directions[direction] = {0, 1}
         * East,  direction = 1, directions[direction] = {1, 0}
         * South, direction = 2, directions[direction] = {0, -1}
         * West,  direction = 3, directions[direction] = {-1, 0}
         */
        int[] commands = new int[]{4, -1, 4, -2, 4};
        int[][] obstacles = new int[][]{{2, 4}};
        int robotSim = week4.robotSim(commands, obstacles);
        int robotSim2 = week4.robotSim2(commands, obstacles);
        System.out.println("robotSim=" + robotSim + "  " + "robotSim2=" + robotSim2);

        int[] jumpNums1 = new int[]{2, 3, 1, 1, 4};
        int[] jumpNums2 = new int[]{3, 2, 1, 0, 4};
        boolean canJump31 = week4.canJump3(jumpNums1);
        boolean canJump32 = week4.canJump3(jumpNums2);
        boolean canJump41 = week4.canJump4(jumpNums1);
        boolean canJump42 = week4.canJump4(jumpNums2);
        System.out.println("canJump31=" + canJump31 + " " + "canJump32=" + canJump32 + " " + "canJump41=" + canJump41 + " " + "canJump42=" + canJump42);


        int[] prices = {7, 5, 8, 3, 6, 9};
        int maxProfit3 = week4.maxProfit3(prices);
        int maxProfit4 = week4.maxProfit4(prices);
        int maxProfit2 = week4.maxProfit2(prices);
        int maxProfit7 = week4.maxProfit7(prices);
        System.out.println(maxProfit3);
        System.out.println(maxProfit4);
        System.out.println(maxProfit2);
        System.out.println(maxProfit7);

        System.out.println("--------------week6--------------");
        Week6 week6 = new Week6();
        int uniquePaths = week6.uniquePaths(4, 3);
        int uniquePaths2 = week6.uniquePaths2(4, 3);
        int uniquePaths3 = week6.uniquePaths3(4, 3);
        int uniquePaths4 = week6.uniquePaths4(4, 3);
        System.out.println("uniquePaths="+uniquePaths + "  uniquePaths2=" + uniquePaths2 +  "  uniquePaths3="+ uniquePaths3 +  "  uniquePaths4=" + uniquePaths4);

        // longestCommonSubsequence
        String text1 = "abcde";
        String text2 = "ace";
        String text3 = "abc";
        String text4 = "def";
        int lcs1 = week6.longestCommonSubsequence(text1, text2);
        int lcs2 = week6.longestCommonSubsequence(text3, text4);
        int lcs3 = week6.longestCommonSubsequence2(text1, text2);
        int lcs4 = week6.longestCommonSubsequence2(text3, text4);
        int lcs5 = week6.longestCommonSubsequence3(text1, text2);
        int lcs6 = week6.longestCommonSubsequence3(text3, text4);
        System.out.println("lcs1=" + lcs1 + "   lcs2=" + lcs2 + "   lcs3=" + lcs3 + "   lcs4=" + lcs4 + "   lcs5=" + lcs5 + "   lcs6=" + lcs6);

        //triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
        List<List<Integer>> triangleList = new ArrayList<List<Integer>>();
        List<Integer> temp1 = new ArrayList<>();
        List<Integer> temp2 = new ArrayList<>();
        List<Integer> temp3 = new ArrayList<>();
        List<Integer> temp4 = new ArrayList<>();
        temp1.add(2);
        temp2.add(3);
        temp2.add(4);
        temp3.add(6);
        temp3.add(5);
        temp3.add(7);
        temp4.add(4);
        temp4.add(1);
        temp4.add(8);
        temp4.add(3);
        triangleList.add(temp1);
        triangleList.add(temp2);
        triangleList.add(temp3);
        triangleList.add(temp4);
        int minimumTotal3 = week6.minimumTotal3(triangleList);
        int minimumTotal4 = week6.minimumTotal4(triangleList);
        System.out.println("minimumTotal3=" + minimumTotal3 + " minimumTotal4=" + minimumTotal4);


        int[] nums6 = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
        int maxSubArray2 = week6.maxSubArray2(nums6);
        int maxSubArray3 = week6.maxSubArray3(nums6);
        System.out.println("maxSubArray2=" + maxSubArray2 + "   maxSubArray3=" + maxSubArray3);

        // 打家劫舍
        int[] numsRob = {1,2,3,1};
        int rob1 = week6.rob(numsRob);
        int rob2 = week6.rob2(numsRob);
        System.out.println("rob1=" + rob1 + "   rob2=" + rob2);

        // 买卖股票的最佳时机
        int[] prices6 = {7, 1, 5, 3, 6, 4};
        int maxProfit61 = week6.maxProfit(prices6);
        int maxProfit62 = week6.maxProfit2(prices6);
        int maxProfit63 = week6.maxProfit3(prices6);
        System.out.println("maxProfit61=" + maxProfit61 + " maxProfit62=" + maxProfit62 + " maxProfit63=" + maxProfit63);

        // 买卖股票的最佳时机 II （亚马逊、字节跳动、微软在半年内面试中考过）
        int[] prices6_2 = {7,6,4,3,1};
        int maxProfit21 = week6.maxProfit21(prices6);
        int maxProfit22 = week6.maxProfit22(prices6);
        int maxProfit23 = week6.maxProfit23(prices6);
        int maxProfit26 = week6.maxProfit26(prices6);
        int maxProfit27 = week6.maxProfit27(prices6);
        int maxProfit28 = week6.maxProfit28(prices6);
        System.out.println("maxProfit21=" + maxProfit21 + " maxProfit22=" + maxProfit22 + " maxProfit23=" + maxProfit23 + " maxProfit26=" + maxProfit26 + " maxProfit27=" + maxProfit27 + " maxProfit28=" + maxProfit28);

        int[] prices6_3 = {7,6,4,3,1};
        int maxProfit31 = week6.maxProfit31(prices6_3);
        int maxProfit32 = week6.maxProfit32(prices6_3);
        int maxProfit33 = week6.maxProfit33(prices6_3);
        int maxProfit34 = week6.maxProfit34(prices6_3);
        int maxProfit35 = week6.maxProfit35(prices6_3);
        int maxProfit36 = week6.maxProfit36(prices6_3);
        System.out.println("maxProfit31=" + maxProfit31 + " maxProfit32=" + maxProfit32 + " maxProfit33=" + maxProfit33 + " maxProfit34=" + maxProfit34 + " maxProfit35=" + maxProfit35 + " maxProfit36=" + maxProfit36);

        // 621. 任务调度器（Facebook 在半年内面试中常考）
        char[] tasks = {'A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'D', 'E', 'F', 'G'};
        int leastInterval = week6.leastInterval(tasks, 2);
        int leastInterval2 = week6.leastInterval2(tasks, 2);
        System.out.println("leastInterval=" + leastInterval + "   leastInterval2=" + leastInterval2);

        // 647. 回文子串    https://leetcode-cn.com/problems/palindromic-substrings/solution/647-hui-wen-zi-chuan-dong-tai-gui-hua-fang-shi-qiu/
        int countSubstrings = week6.countSubstrings("aabaa");
        int countSubstrings2 = week6.countSubstrings2("aabaa");
        int countSubstrings3 = week6.countSubstrings3("aabaa");
        int countSubstrings4 = week6.countSubstrings4("aabaa");
        int countSubstrings5 = week6.countSubstrings5("aabaa");
        System.out.println("countSubstrings=" + countSubstrings + "   countSubstrings2=" + countSubstrings2 + "   countSubstrings3=" + countSubstrings3 + "   countSubstrings4=" + countSubstrings4 + "   countSubstrings5=" + countSubstrings5);

        System.out.println('D' - 'A');// 输出:3

        // 5. 最长回文子串
        String string_longestPalindrome = "babad";
        String longestPalindrome = week6.longestPalindrome(string_longestPalindrome);
        String longestPalindrome2 = week6.longestPalindrome2(string_longestPalindrome);
        String longestPalindrome3 = week6.longestPalindrome3(string_longestPalindrome);
        String longestPalindrome4 = week6.longestPalindrome4(string_longestPalindrome);
        String longestPalindrome5 = week6.longestPalindrome5(string_longestPalindrome);
        String longestPalindrome6 = week6.longestPalindrome6(string_longestPalindrome);
        System.out.println("longestPalindrome=" + longestPalindrome + "   longestPalindrome2=" + longestPalindrome2 + "   longestPalindrome3=" + longestPalindrome3 + "   longestPalindrome4=" + longestPalindrome4 + "   longestPalindrome5=" + longestPalindrome5 + "   longestPalindrome6=" + longestPalindrome6);


    }
}
