package algorithm;

/**
 * @ClassName Week9
 * @Description
 * @Author Administrator
 * @Date 2021/1/11  9:38
 * @Version 1.0
 **/
public class Week9 {

    /**
     * 300. 最长上升子序列（字节跳动、亚马逊、微软在半年内面试中考过）
     * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
     * 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列
     * 链接：https://leetcode-cn.com/problems/longest-increasing-subsequence
     * *****************************************************************************************************************
     * 示例 1：
     * 输入：nums = [10,9,2,5,3,7,101,18]
     * 输出：4
     * 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4
     * *****************************************************************************************************************
     * https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/zui-chang-shang-sheng-zi-xu-lie-by-leetcode-soluti/
     * 方法一：动态规划
     */
    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int maxans = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            maxans = Math.max(maxans, dp[i]);
        }
        return maxans;
    }

    /**
     * 方法二：贪心 + 二分查找
     */
    public int lengthOfLIS2(int[] nums) {
        int len = 1, n = nums.length;
        if (n == 0) {
            return 0;
        }
        int[] d = new int[n + 1];
        d[len] = nums[0];
        for (int i = 1; i < n; ++i) {
            if (nums[i] > d[len]) {
                d[++len] = nums[i];
            } else {
                int l = 1, r = len, pos = 0; // 如果找不到说明所有的数都比 nums[i] 大，此时要更新 d[1]，所以这里将 pos 设为 0
                while (l <= r) {
                    int mid = (l + r) >> 1;
                    if (d[mid] < nums[i]) {
                        pos = mid;
                        l = mid + 1;
                    } else {
                        r = mid - 1;
                    }
                }
                d[pos + 1] = nums[i];
            }
        }
        return len;
    }

    /**
     * 91. 解码方法（Facebook、亚马逊、字节跳动在半年内面试中考过）
     * 一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：
     *
     * 'A' -> 1
     * 'B' -> 2
     * ...
     * 'Z' -> 26
     * 要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。
     * 例如，"111" 可以将 "1" 中的每个 "1" 映射为 "A" ，从而得到 "AAA" ，或者可以将 "11" 和 "1"（分别为 "K" 和 "A" ）映射为 "KA" 。注意，"06" 不能映射为 "F" ，因为 "6" 和 "06" 不同。
     * 给你一个只含数字的 非空 字符串 num ，请计算并返回 解码 方法的 总数 。
     * 题目数据保证答案肯定是一个 32 位 的整数。
     * 链接：https://leetcode-cn.com/problems/decode-ways
     * *****************************************************************************************************************
     * 动态规划（Java、Python）
     * 一句话题解：根据一个字符串结尾的两个字符（暂不讨论边界情况），推导状态转移方程。
     *
     * 这一类问题，问方案数，但不问具体方案的，可以考虑使用「动态规划」完成；
     * 「动态规划」处理字符串问题的思想是：从一个空串开始，一点一点得到更大规模的问题的解。
     * 说明：这个问题有点像 「力扣」第 70 题：爬楼梯，背后的思想是「分类计数加法原理」和「分步计数乘法原理」。
     *
     * 方法一：动态规划
     *
     * 第 1 步：定义状态
     *
     * 既然结尾的字符很重要，在定义状态的时候可以这样定义：
     *
     * dp[i]：以 s[i] 结尾的前缀子串有多少种解码方法。
     *
     * 第 2 步：推导状态转移方程
     *
     * 根据题意：
     *
     * 如果 s[i] == '0' ，字符 s[i] 就不能单独解码，所以当 s[i] != '0' 时，dp[i] = dp[i - 1] * 1。
     * 说明：为了得到长度为 i + 1 的前缀子串的解码个数，需要先得到长度为 i 的解码个数，再对 s[i] 单独解码，这里分了两步，根据「分步计数原理」，用乘法。这里的 1 表示乘法单位，语义上表示 s[i] 只有 1 种编码。
     *
     * 如果当前字符和它前一个字符，能够解码，即 10 <= int(s[i - 1..i]) <= 26，即 dp[i] += dp[i - 2] * 1；
     * 说明：不同的解码方法，使用「加法」，理论依据是「分类计数的加法原理」，所以这里用 +=。
     *
     * 注意：状态转移方程里出现了下标 i - 2，需要单独处理（如何单独处理，需要耐心调试）。
     *
     * 第 3 步：初始化
     *
     * 如果首字符为 0 ，一定解码不了，可以直接返回 0，非零情况下，dp[0] = 1；
     * 第 4 步：考虑输出
     *
     * 输出是 dp[len - 1]，符合原始问题。
     *
     * 第 5 步：考虑优化空间
     *
     * 这里当前状态值与前面两个状态有关，因此可以使用三个变量滚动计算。但空间资源一般来说不紧张，不是优化的方向，故不考虑。
     *
     * 参考代码 1：
     * 链接：https://leetcode-cn.com/problems/decode-ways/solution/dong-tai-gui-hua-java-python-by-liweiwei1419/
     */
    public int numDecodings(String s) {
        int len = s.length();
        if (len == 0) {
            return 0;
        }

        // dp[i] 以 s[i] 结尾的前缀子串有多少种解码方法
        // dp[i] = dp[i - 1] * 1 if s[i] != '0'
        // dp[i] += dp[i - 2] * 1 if  10 <= int(s[i - 1..i]) <= 26
        int[] dp = new int[len];

        char[] charArray = s.toCharArray();
        if (charArray[0] == '0') {
            return 0;
        }
        dp[0] = 1;

        for (int i = 1; i < len; i++) {
            if (charArray[i] != '0') {
                dp[i] = dp[i - 1];
            }

            int num = 10 * (charArray[i - 1] - '0') + (charArray[i] - '0');
            if (num >= 10 && num <= 26) {
                if (i == 1) {
                    dp[i]++;
                } else {
                    dp[i] += dp[i - 2];
                }
            }
        }
        return dp[len - 1];
    }

    /**
     * 方法二：基于方法一修改状态定义
     *
     * 这里在 i == 1 的时候需要多做一次判断，而这种情况比较特殊，为了避免每次都做判断，可以把状态数组多设置一位。为此修改状态定义，与此同时，状态转移方程也需要做一点点调整。
     *
     * dp[i] 定义成长度为 i 的前缀子串有多少种解码方法（以 s[i - 1] 结尾的前缀子串有多少种解法方法）；
     *
     * 状态转移方程：分类讨论，注意字符的下标和状态数组的有 1 个下标的偏移。
     *
     * 当 s[i] != '0' 时，dp[i + 1] = dp[i]；
     * 当 10 <= s[i - 1..i] <= 26 时，dp[i + 1] += dp[i - 1]。
     * 初始化：dp[0] = 1，意义：0 个字符的解码方法为 1 种。如果实在不想深究这里的语义，也没有关系，dp[0] 的值是用于后面状态值参考的，在 i == 1 的时候，dp[i + 1] += dp[i - 1] 即 dp[2] += dp[0] ，这里就需要 dp[0] = 1 。
     *
     * 输出：dp[len]
     *
     * 参考代码 2：
     *
     * 作者：liweiwei1419
     * 链接：https://leetcode-cn.com/problems/decode-ways/solution/dong-tai-gui-hua-java-python-by-liweiwei1419/
     * 来源：力扣（LeetCode）
     * 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
     */
    public int numDecodings2(String s) {
        int len = s.length();
        if (len == 0) {
            return 0;
        }

        // dp[i] 以 s[i - 1] 结尾的前缀子串有多少种解法方法
        // dp[i] = dp[i - 1] * 1 if nums[i - 1] != '0'
        // dp[i] += dp[i - 2] * 1 if  10 <= int(s[i - 2..i - 1]) <= 26

        int[] dp = new int[len + 1];
        dp[0] = 1;
        char[] charArray = s.toCharArray();
        if (charArray[0] == '0') {
            return 0;
        }
        dp[1] = 1;

        for (int i = 1; i < len; i++) {
            if (charArray[i] != '0') {
                dp[i + 1] = dp[i];
            }

            int num = 10 * (charArray[i - 1] - '0') + (charArray[i] - '0');
            if (num >= 10 && num <= 26) {
                dp[i + 1] += dp[i - 1];
            }
        }
        return dp[len];
    }

    /**
     * 正向填表：
     *
     * class Solution:
     *     def numDecodings(self, s):
     *         if not s or s[0] == '0': return 0
     *         n = len(s)
     *         dp = [0] * (n + 1)
     *         dp[0] = dp[1] = 1  # 处理‘10’这样的情况，首位初始化为1
     *
     *         for i in range(2, n + 1):
     *             if s[i-1] != '0': dp[i] += dp[i-1]
     *             if 10 <= int(s[i-2:i]) <= 26: dp[i] += dp[i-2]
     *
     *         return dp[-1]
     * 反向填表：
     *
     * class Solution:
     *     def numDecodings(self, s):
     *         if not s: return 0
     *         n = len(s)
     *         dp = [0] * (n + 1)
     *         dp[n] = 1
     *         if s[n-1] != '0': dp[n-1] = 1
     *
     *         for i in range(n-2, -1, -1):
     *             if s[i] == '0': continue
     *             dp[i] += dp[i+1]
     *             if int(s[i:i+2]) <= 26: dp[i] += dp[i+2]
     *         return dp[0]
     */

}
