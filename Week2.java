package algorithm;


import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @ClassName Week2
 * @Description
 * @Author Administrator
 * @Date 2020/11/24  5:09
 * @Version 1.0
 **/
public class Week2 {

    /**
     *  第2周 第5课 | 哈希表、映射、集合
     *  第2周 第6课 | 树、二叉树、二叉搜索树
     *  第2周 第6课 | 堆和二叉堆、图
     * *****************************************************************************************************************
     *  第2周 第5课 | 哈希表、映射、集合
     *  有效的字母异位词（亚马逊、Facebook、谷歌在半年内面试中考过）
     *  字母异位词分组（亚马逊在半年内面试中常考）
     *  两数之和（亚马逊、字节跳动、谷歌、Facebook、苹果、微软、腾讯在半年内面试中常考）
     */

    /**
     * 242. 有效的字母异位词（亚马逊、Facebook、谷歌在半年内面试中考过）
     * anagram : 相同字母异序词.
     * 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
     * 示例 1:
     *
     * 输入: s = "anagram", t = "nagaram"
     * 输出: true
     * 示例 2:
     *
     * 输入: s = "rat", t = "car"
     * 输出: false
     * *****************************************************************************************************************
     * 方法一：排序
     * 方法二：哈希表
     * https://leetcode-cn.com/problems/valid-anagram/solution/you-xiao-de-zi-mu-yi-wei-ci-by-leetcode-solution/
     */
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length())
            return false;
        int[] alpha = new int[26];
        for (int i = 0; i < s.length(); i++) {
            alpha[s.charAt(i) - 'a']++;
            alpha[t.charAt(i) - 'a']--;
        }

        // return Arrays.stream(alpha).noneMatch(num -> num != 0); //合并下面操作
        for (int i = 0; i < 26; i++) {
            if (alpha[i] != 0)
                return false;
        }
        return true;
    }

    public boolean isAnagram2(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        char[] s1 = s.toCharArray();
        char[] t2 = t.toCharArray();
        Arrays.sort(s1);
        Arrays.sort(t2);
        for (int i = 0; i < s1.length; i++) {
            if (s1[i] != t2[i]) {
                return false;
            }
        }
        return true;
    }

    // Valid anagram
    public boolean isAnagram3(String s, String t) {
        if (s.length() != t.length()) return false;
        int[] a = new int[26];
        for (Character c : s.toCharArray()) a[c - 'a']++;
        for (Character c : t.toCharArray()) {
            if (a[c - 'a'] == 0) return false;
            a[c - 'a']--;
        }
        return true;
    }

    public boolean isAnagram4(String s1, String s2) {
        int[] freq = new int[256];
        for (int i = 0; i < s1.length(); i++) freq[s1.charAt(i)]++;
        for (int i = 0; i < s2.length(); i++) if (--freq[s2.charAt(i)] < 0) return false;
        return s1.length() == s2.length();
    }

    public boolean isAnagram5(String s, String t) {
        char[] sChar = s.toCharArray();
        char[] tChar = t.toCharArray();
        Arrays.sort(sChar);
        Arrays.sort(tChar);
        return Arrays.equals(sChar, tChar);
    }

    /**
     * 49. 字母异位词分组（亚马逊在半年内面试中常考）
     * 给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
     * 示例:
     *
     * 输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
     * 输出:
     * [
     *   ["ate","eat","tea"],
     *   ["nat","tan"],
     *   ["bat"]
     * ]
     * *****************************************************************************************************************
     * 思路:分组问题要有Key值来作为组的唯一识别标志;关键步骤是找到Key
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, ArrayList<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] ch = s.toCharArray();
            Arrays.sort(ch);
            String key = String.valueOf(ch);
            if (!map.containsKey(key))
                map.put(key, new ArrayList<>());
            map.get(key).add(s);
        }
        return new ArrayList<>(map.values());
    }

    // 字母异位词分组 2ed 方法
    public List<List<String>> groupAnagrams2(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            map.computeIfAbsent(new String(chars), k -> new ArrayList<>()).add(str);
        }
        return new ArrayList<>(map.values());
    }

    // Group Anagrams:
    public List<List<String>> groupAnagrams3(String[] strs) {
        List<List<String>> res = new ArrayList<>();
        HashMap<String, List<String>> map = new HashMap<>();

        Arrays.sort(strs);
        for (int i = 0; i < strs.length; i++) {
            String temp = strs[i];
            char[] ch = temp.toCharArray();
            Arrays.sort(ch);
            if (map.containsKey(String.valueOf(ch))) {
                map.get(String.valueOf(ch)).add(strs[i]);
            } else {
                List<String> each = new ArrayList<>();
                each.add(strs[i]);
                map.put(String.valueOf(ch), each);
            }
        }
        for (List<String> item : map.values()) {
            res.add(item);
        }
        return res;
    }

    /**
     * 看我一句话 AC 字母异位词分组！
     * 题目描述
     * 给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
     *
     * 思路解析
     * 方法一：排序
     * 字母相同，但排列不同的字符串，排序后都一定是相同的。因为每种字母的个数都是相同的，那么排序后的字符串就一定是相同的。
     *
     * 这里可以利用 stream 的 groupingBy 算子实现直接返回结果：
     *
     * 注意 groupingBy 算子计算完以后，返回的是一个 Map<String, List<String>>，map 的键是每种排序后的字符串，值是聚合的原始字符串，我们只关心值，所以我们最后 new ArrayList<>(map.values())。
     *
     * 作者：sweetiee
     * 链接：https://leetcode-cn.com/problems/group-anagrams/solution/kan-wo-yi-ju-hua-ac-zi-mu-yi-wei-ci-fen-yrnis/
     */
    public List<List<String>> groupAnagrams4(String[] strs) {
        return new ArrayList<>(Arrays.stream(strs).collect(Collectors.groupingBy(str -> {
                    // 返回 str 排序后的结果。
                    // 按排序后的结果来grouping by，算子类似于 sql 里的 group by。
                    char[] array = str.toCharArray();
                    Arrays.sort(array);
                    return new String(array);
                })).values());
    }

    /**
     * 方法一之狂拽炫技
     * 其实这里由于需要计算排序后的字符串，导致使用了多行代码，为了狂 (sang) 拽 (xin) 炫 (bing) 技 (kuang)，还可以利用 stream 对字符串排序，这样只需要一行代码即可。代码如下：
     *
     * 作者：sweetiee
     * 链接：https://leetcode-cn.com/problems/group-anagrams/solution/kan-wo-yi-ju-hua-ac-zi-mu-yi-wei-ci-fen-yrnis/
     */
    public List<List<String>> groupAnagrams5(String[] strs) {
        // str -> intstream -> sort -> collect by StringBuilder
        return new ArrayList<>(Arrays.stream(strs).collect(Collectors.groupingBy(str -> str.chars().sorted().collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append).toString())).values());
    }
    /**
     * 还有一种写法，不过由于会 split 字符串，会慢很多（但是短啊！）：
     * *****************************************************************************************************************
     * 设 n 是数组长度，k 是字符串最大长度。
     *
     * 时间复杂度：O(nklogk)。每个字符串排序，时间复杂度 O(klogk)，排序 n 个，就是 O(nklogk)。groupingBy 的时间复杂度是 O(n) 的，所以整体是 O(nklogk)。
     * 空间复杂度： O(nk)。 groupingBy 后产生的 HashMap 会存所有的字符串。
     *
     * 作者：sweetiee
     * 链接：https://leetcode-cn.com/problems/group-anagrams/solution/kan-wo-yi-ju-hua-ac-zi-mu-yi-wei-ci-fen-yrnis/
     */
    public List<List<String>> groupAnagrams6(String[] strs) {
        // str -> split -> stream -> sort -> join
        return new ArrayList<>(Arrays.stream(strs).collect(Collectors.groupingBy(str -> Stream.of(str.split("")).sorted().collect(Collectors.joining()))).values());
    }

    /**
     * 方法二：计数
     * 对每个字符串计数得到该字符串的计数数组，对于计数数组相同的字符串，就互为异位词。
     * 因为数组类型没有重写 hashcode() 和 equals() 方法，因此不能直接作为 HashMap 的 Key 进行聚合，那么我们就 把这个数组手动编码变成字符串就行了。
     * 比如将 [b,a,a,a,b,c] 编码成 a3b2c1，使用编码后的字符串作为 HashMap 的 Key 进行聚合。
     * *****************************************************************************************************************
     * 设 n 是数组长度，k 是字符串最大长度。
     *
     * 时间复杂度： O(nk)。 每个字符串计数再编码，由于题目说明是小写字母，所以是 O(n(k + 26))，常数忽略后就是 O(nk)。
     * 空间复杂度： O(nk)。 groupingBy 后产生的 HashMap 会存所有的字符串。
     *
     * 作者：sweetiee
     * 链接：https://leetcode-cn.com/problems/group-anagrams/solution/kan-wo-yi-ju-hua-ac-zi-mu-yi-wei-ci-fen-yrnis/
     */
    public List<List<String>> groupAnagrams7(String[] strs) {
        return new ArrayList<>(Arrays.stream(strs).collect(Collectors.groupingBy(str -> {
            int[] counter = new int[26];
            for (int i = 0; i < str.length(); i++) {
                counter[str.charAt(i) - 'a']++;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 26; i++) {
                // 这里的 if 是可省略的，但是加上 if 以后，生成的 sb 更短，后续 groupingBy 会更快。
                if (counter[i] != 0) {
                    sb.append((char) ('a' + i));
                    sb.append(counter[i]);
                }
            }
            return sb.toString();
        })).values());
    }

    /**
     * 第2周 第6课 | 树、二叉树、二叉搜索树
     * *****************************************************************************************************************
     * 二叉树的中序遍历（亚马逊、微软、字节跳动在半年内面试中考过）
     * 二叉树的前序遍历（谷歌、微软、字节跳动在半年内面试中考过）
     *  N 叉树的后序遍历（亚马逊在半年内面试中考过）
     *  N 叉树的前序遍历（亚马逊在半年内面试中考过）
     *  N 叉树的层序遍历
     * *****************************************************************************************************************
     * 144. 二叉树的前序遍历（谷歌、微软、字节跳动在半年内面试中考过）
     * 给你二叉树的根节点 root ，返回它节点值的 前序 遍历。
     *
     * 示例 1：
     * 输入：root = [1,null,2,3]
     * 输出：[1,2,3]
     *
     * 示例 2：
     * 输入：root = []
     * 输出：[]
     *
     * 示例 3：
     * 输入：root = [1]
     * 输出：[1]
     *
     * 示例 4：
     * 输入：root = [1,2]
     * 输出：[1,2]
     *
     * 示例 5：
     * 输入：root = [1,null,2]
     * 输出：[1,2]
     * *****************************************************************************************************************
     * 递归
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null) return list;
        dfs(list, root);
        return list;
    }

    public void dfs(List<Integer> list, TreeNode root) {
        if (root == null) return;
        list.add(root.val);
        dfs(list, root.left);
        dfs(list, root.right);
    }

    /**
     * 图解 二叉树的四种遍历
     * LeetCode 题目中，二叉树的遍历方式是最基本，也是最重要的一类题目，我们将从「前序」、「中序」、「后序」、「层序」四种遍历方式出发，总结他们的递归和迭代解法。
     * *****************************************************************************************************************
     * 1. 相关题目
     * 这里是 4 道相关题目：
     * 144.二叉树的前序遍历
     * 94. 二叉树的中序遍历
     * 145.二叉树的后序遍历
     * 102.二叉树的层序遍历
     * *****************************************************************************************************************
     * 2. 基本概念
     * 要解决这四道题目，最基本的前提是要了解什么是二叉树，以及二叉树的遍历方式。如果你已经有所了解，则可以直接查看下一节的内容。
     * 二叉树
     * 首先，二叉树是一种「数据结构」，详细的介绍可以通过 「探索」卡片 来进行学习。简单来说，就是一个包含节点，以及它的左右孩子的一种数据结构。
     * 见图（底部链接）
     * 遍历方式
     * 如果对每一个节点进行编号，你会用什么方式去遍历每个节点呢？
     * 见图（底部链接）
     * 如果你按照 根节点 -> 左孩子 -> 右孩子 的方式遍历，即「先序遍历」，每次先遍历根节点，遍历结果为 1 2 4 5 3 6 7；
     *
     * 同理，如果你按照 左孩子 -> 根节点 -> 右孩子 的方式遍历，即「中序序遍历」，遍历结果为 4 2 5 1 6 3 7；
     *
     * 如果你按照 左孩子 -> 右孩子 -> 根节点 的方式遍历，即「后序序遍历」，遍历结果为 4 5 2 6 7 3 1；
     *
     * 最后，层次遍历就是按照每一层从左向右的方式进行遍历，遍历结果为 1 2 3 4 5 6 7。
     *
     * 3. 题目解析
     * 这四道题目描述是相似的，就是给定一个二叉树，让我们使用一个数组来返回遍历结果，首先来看递归解法。
     *
     * 3.1 递归解法
     * 由于层次遍历的递归解法不是主流，因此只介绍前三种的递归解法。它们的模板相对比较固定，一般都会新增一个 dfs 函数：
     * 对于前序、中序和后序遍历，只需将递归函数里的 res.append(root.val) 放在不同位置即可，然后调用这个递归函数就可以了，代码完全一样。
     *
     * 1. 前序遍历
     *
     * 2. 中序遍历
     *
     * 3. 后序遍历
     *
     * 一样的代码，稍微调用一下位置就可以，如此固定的套路，使得只掌握递归解法并不足以令面试官信服。
     * 因此我们有必要再掌握迭代解法，同时也会加深我们对数据结构的理解。
     *
     * 3.2 迭代解法
     * a. 二叉树的前序遍历
     * LeetCode 题目： 144.二叉树的前序遍历
     * 常规解法
     * 我们使用栈来进行迭代，过程如下：
     * 初始化栈，并将根节点入栈；
     * 当栈不为空时：
     * 弹出栈顶元素 node，并将值添加到结果中；
     * 如果 node 的右子树非空，将右子树入栈；
     * 如果 node 的左子树非空，将左子树入栈；
     * 由于栈是“先进后出”的顺序，所以入栈时先将右子树入栈，这样使得前序遍历结果为 “根->左->右”的顺序。
     * 参考代码如下：
     * 模板解法
     * 当然，你也可以直接启动“僵尸”模式，套用迭代的模板来一波“真香操作”。
     * 模板解法的思路稍有不同，它先将根节点 cur 和所有的左孩子入栈并加入结果中，直至 cur 为空，用一个 while 循环实现：
     * 然后，每弹出一个栈顶元素 tmp，就到达它的右孩子，再将这个节点当作 cur 重新按上面的步骤来一遍，直至栈为空。这里又需要一个 while 循环。
     * 参考代码如下：
     * b. 二叉树的中序遍历
     * LeetCode 题目：94. 二叉树的中序遍历
     * 模板解法
     * 和前序遍历的代码完全相同，只是在出栈的时候才将节点 tmp 的值加入到结果中。
     * c. 二叉树的后序遍历
     * LeetCode 题目：145. 二叉树的后序遍历
     * 模板解法
     * 继续按照上面的思想，这次我们反着思考，节点 cur 先到达最右端的叶子节点并将路径上的节点入栈；
     * 然后每次从栈中弹出一个元素后，cur 到达它的左孩子，并将左孩子看作 cur 继续执行上面的步骤。
     * 最后将结果反向输出即可。参考代码如下：
     *
     * 然而，后序遍历采用模板解法并没有按照真实的栈操作，而是利用了结果的特点反向输出，不免显得技术含量不足。
     * 因此掌握标准的栈操作解法是必要的。
     * 常规解法
     * 类比前序遍历的常规解法，我们只需要将输出的“根 -> 左 -> 右”的顺序改为“左 -> 右 -> 根”就可以了。
     * 如何实现呢？这里右一个小技巧，我们入栈时额外加入一个标识，比如这里使用 flag = 0；
     *
     * 然后每次从栈中弹出元素时，如果 flag 为 0,则需要将 flag 变为 1 并连同该节点再次入栈，只有当 flag 为 1 时才可将该节点加入到结果中。
     * 参考代码如下：
     *
     * 4. 二叉树的层次遍历
     * LeetCode 题目：102. 二叉树的层序遍历
     * 二叉树的层次遍历的迭代方法与前面不用，因为前面的都采用了深度优先搜索的方式，而层次遍历使用了广度优先搜索，广度优先搜索主要使用队列实现，也就不能使用前面的模板解法了。
     * 广度优先搜索的步骤为：
     * ①初始化队列 q，并将根节点 root 加入到队列中；
     * ②当队列不为空时：
     * 队列中弹出节点 node，加入到结果中；
     * 如果左子树非空，左子树加入队列；
     * 如果右子树非空，右子树加入队列；
     * 由于题目要求每一层保存在一个子数组中，所以我们额外加入了 level 保存每层的遍历结果，并使用 for 循环来实现。
     *
     *
     * 4. 总结
     * 总结一下，在二叉树的前序、中序、后序遍历中，递归实现的伪代码为：
     * 链接：https://leetcode-cn.com/problems/binary-tree-preorder-traversal/solution/tu-jie-er-cha-shu-de-si-chong-bian-li-by-z1m/
     * 迭代实现的伪代码为：
     * 链接：https://leetcode-cn.com/problems/binary-tree-preorder-traversal/solution/tu-jie-er-cha-shu-de-si-chong-bian-li-by-z1m/
     * 掌握了以上基本的遍历方式，对待更多的进阶题目就游刃有余了。
     */

    /**
     * 94. 二叉树的中序遍历（亚马逊、微软、字节跳动在半年内面试中考过）
     * 给定一个二叉树的根节点 root ，返回它的 中序 遍历。
     * *****************************************************************************************************************
     * 示例 1：
     * 输入：root = [1,null,2,3]
     * 输出：[1,3,2]
     * 示例 2：
     *
     * 输入：root = []
     * 输出：[]
     * 示例 3：
     *
     * 输入：root = [1]
     * 输出：[1]
     * *****************************************************************************************************************
     * 动画演示+三种实现 94. 二叉树的中序遍历
     * 解题思路：
     * 递归实现
     * 递归遍历太简单了
     * 前序遍历：打印 - 左 - 右
     * 中序遍历：左 - 打印 - 右
     * 后序遍历：左 - 右 - 打印
     * 题目要求的是中序遍历，那就按照 左-打印-右这种顺序遍历树就可以了，递归函数实现
     * 终止条件：当前节点为空时
     * 函数内：递归的调用左节点，打印当前节点，再递归调用右节点
     * 链接：https://leetcode-cn.com/problems/binary-tree-inorder-traversal/solution/dong-hua-yan-shi-94-er-cha-shu-de-zhong-xu-bian-li/
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        dfsInorderTraversal(res, root);
        return res;
    }

    void dfsInorderTraversal(List<Integer> res, TreeNode root) {
        if (root == null) {
            return;
        }
        //按照 左-打印-右的方式遍历
        dfsInorderTraversal(res, root.left);
        res.add(root.val);
        dfsInorderTraversal(res, root.right);
    }

    /**
     * 589. N 叉树的前序遍历（亚马逊在半年内面试中考过）【递归】【迭代】
     * 给定一个 N 叉树，返回其节点值的 前序遍历 。
     * N 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 null 分隔（请参见示例）。
     * 进阶：
     * 递归法很简单，你可以使用迭代法完成此题吗?
     * *****************************************************************************************************************
     * 示例 1：
     * 输入：root = [1,null,3,2,4,null,5,6]
     * 输出：[1,3,5,6,2,4]
     *
     * 示例 2：
     * 输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
     * 输出：[1,2,3,6,7,11,14,4,8,12,5,9,13,10]
     * *****************************************************************************************************************
     * 589. N 叉树的前序遍历【递归】【迭代】
     * https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/solution/589-n-cha-shu-de-qian-xu-bian-li-di-gui-x2n16/
     *
     * 1.递归版
     */
    public List<Integer> preorder(Node root) {
        List<Integer> list = new ArrayList<>();
        helper(root, list);
        return list;
    }

    public void helper(Node root, List<Integer> list) {
        if (root == null) return;
        list.add(root.val);
        for (Node child : root.children) {
            helper(child, list);
        }
    }
    /**
     * 2.迭代版
     */
    public List<Integer> preorder2(Node root) {
        List<Integer> list = new ArrayList<>();
        Deque<Node> stack = new LinkedList<>();
        stack.push(root);
        if (root == null) return list;
        while (!stack.isEmpty()) {
            //当前栈顶节点出栈
            Node node = stack.pop();
            //将值加入列表
            list.add(node.val);
            int size = node.children.size();
            //将当前节点的孩子们从右到左入栈
            for (int i = size - 1; i >= 0; i--) {
                stack.push(node.children.get(i));
            }
        }
        return list;
    }

    /**
     * 590. N 叉树的后序遍历（亚马逊在半年内面试中考过）
     * 给定一个 N 叉树，返回其节点值的 后序遍历 。
     * N 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 null 分隔（请参见示例）。
     * 进阶：
     * 递归法很简单，你可以使用迭代法完成此题吗?
     * *****************************************************************************************************************
     * 590. N-ary Tree Postorder Traversal
     * Given the root of an n-ary tree, return the postorder traversal of its nodes' values.
     * Nary-Tree input serialization is represented in their level order traversal. Each group of children is separated by the null value (See examples)
     * *****************************************************************************************************************
     * https://leetcode.com/problems/n-ary-tree-postorder-traversal/discuss/147959/Java-Iterative-and-Recursive-Solutions
     *
     * Java Iterative and Recursive Solutions
     *
     * Iterative
     */
    public List<Integer> postorder(Node root) {
        List<Integer> list = new ArrayList<>();
        if (root == null) return list;

        Stack<Node> stack = new Stack<>();
        stack.add(root);

        while (!stack.isEmpty()) {
            root = stack.pop();
            list.add(root.val);
            for (Node node : root.children)
                stack.add(node);
        }
        Collections.reverse(list);
        return list;
    }

    /**
     * Recursive
     * *****************************************************************************************************************
     * avoiding global variable by below:
     */
    public List<Integer> postorder2(Node root) {
        List<Integer> list = new LinkedList<>();
        if (root == null)
            return list;
        for (Node child : root.children) {
            list.addAll(postorder2(child));
        }
        list.add(root.val);
        return list;
    }

    /**
     * Easier solution where i used LinkedList instead of ArrayList and keep adding at head.
     */
    public List<Integer> postorder3(Node root) {
        LinkedList<Integer> list = new LinkedList<Integer>();
        if (root == null)
            return list;
        Stack<Node> stack = new Stack<Node>();
        stack.push(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            List<Node> child = node.children;
            list.addFirst(node.val);
            for (Node no : child) {
                stack.push(no);
            }
        }
        return list;
    }

    /**
     * 429. N 叉树的层序遍历
     * 给定一个 N 叉树，返回其节点值的层序遍历。（即从左到右，逐层遍历）。
     * 树的序列化输入是用层序遍历，每组子节点都由 null 值分隔（参见示例）。
     * *****************************************************************************************************************
     * 本题见官网题解（优质）
     *
     * 方法一：利用队列实现广度优先搜索
     * 我们要构造一个 sub-lists 列表，其中每个 sub-list 是树中一行的值。行应该按从上到下的顺序排列。
     * 因为我们从根节点开始遍历树，然后向下搜索最接近根节点的节点，这是广度优先搜索。我们使用队列来进行广度优先搜索，队列具有先进先出的特性。
     * 在这里使用栈是错误的选择，栈应用于深度优先搜索。
     * 让我们在树上使用基于队列的遍历算法，看看它的作用。这是你应该记住的一个基本算法。
     *
     * List<Integer> values = new ArrayList<>();
     * Queue<Node> queue = new LinkedList<>();
     * queue.add(root);
     * while (!queue.isEmpty()) {
     *     Node nextNode = queue.remove();
     *     values.add(nextNode.val);
     *     for (Node child : nextNode.children) {
     *         queue.add(child);
     *     }
     * }
     *
     * 用一个列表存放节点值，队列存放节点。首先将根节点放到队列中，当队列不为空时，则在队列取出一个节点，并将其子节点添加到队列中。
     * 让我们看看这个算法遍历树时我们得到了什么结果。
     *
     * 我们可以看到它从左到右，并且从上到下顺序遍历节点。下一步，我们将研究如何如何在这个算法的基础上保存每一层的列表。
     * 算法：
     * 上面的基本算法在一定程度上帮助了我们解决这道题目，但是我们还需要保存每一层的列表，并且在根节点为空时正常工作。
     * 再构造下一层的列表时，我们需要创建新的子列表，然后将该层的所有节点的值插入到列表中。一个很好的方法时在 while 循环体开始时记录队列的当前大小 size。然后用另一个循环来处理 size 数量的节点。这样可以保证 while 循环在每一次迭代处理一层。
     * 使用队列十分重要，如果使用 Vector，List，Array 的话，我们删除元素需要 O(n)O(n) 的时间复杂度。而队列删除元素只需要 O(1)O(1) 的时间。
     *
     * 链接：https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/solution/ncha-shu-de-ceng-xu-bian-li-by-leetcode/
     *
     * This code is a modified version of the code posted by
     * #zzzliu on the discussion forums.
     */
    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Node node = queue.poll();
                level.add(node.val);
                queue.addAll(node.children);
            }
            result.add(level);
        }
        return result;
    }

    /**
     * 方法二：简化的广度优先搜索
     * *****************************************************************************************************************
     * This code is a modified version of the code posted by
     * #zzzliu on the discussion forums.
     * 复杂度分析
     * 时间复杂度：O(n)。n 指的是节点的数量。
     * 空间复杂度：O(n)，我们的列表包含所有节点。
     */
    public List<List<Integer>> levelOrder2(Node root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;

        List<Node> previousLayer = Arrays.asList(root);

        while (!previousLayer.isEmpty()) {
            List<Node> currentLayer = new ArrayList<>();
            List<Integer> previousVals = new ArrayList<>();
            for (Node node : previousLayer) {
                previousVals.add(node.val);
                currentLayer.addAll(node.children);
            }
            result.add(previousVals);
            previousLayer = currentLayer;
        }

        return result;
    }

    /**
     * 方法三：递归
     * 我们可以使用递归来解决这个问题，通常我们不能使用递归进行广度优先搜索。
     * 这是因为广度优先搜索基于队列，而递归运行时使用堆栈，适合深度优先搜索。
     * 但是在本题中，我们可以以不同的顺序添加到最终列表中，只要我们知道节点在哪一层并确保在那一层的列表顺序正确就可以了。
     * *****************************************************************************************************************
     * 递归层序遍历，几乎双百
     * 见下链接
     * https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/solution/di-gui-ceng-xu-bian-li-ji-hu-shuang-bai-by-cyingen/
     */
    public List<List<Integer>> levelOrder3(Node root) {
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) return ans;
        leveltravel(ans, 0, root);
        return ans;
    }

    private void leveltravel(List<List<Integer>> ans, int level, Node root) {
        if (root == null) return;
        if (ans.size() < level + 1) ans.add(new ArrayList<Integer>());
        ans.get(level).add(root.val);
        int childnum = root.children.size();
        for (int j = 0; j < childnum; j++)
            leveltravel(ans, level + 1, root.children.get(j));
    }

    /**
     * 第2周 第6课 | 堆和二叉堆、图
     * 1. 堆和二叉堆的实现和特性
     * 2. 实战题目解析：最小的k个数、滑动窗口最大值等问题
     * 3. 图的实现和特性
     * 4. 本周学习问题反馈
     * 5. 本周作业及下周预习
     * *****************************************************************************************************************
     * 1. 堆和二叉堆的实现和特性
     * 参考链接
     * 维基百科：堆（Heap）
     * 堆的实现代码： https://shimo.im/docs/Lw86vJzOGOMpWZz2/
     * 2. 实战题目解析：最小的k个数、滑动窗口最大值等问题
     * 实战例题
     * 最小的 k 个数（字节跳动在半年内面试中考过）
     * 滑动窗口最大值（亚马逊在半年内面试中常考）
     * 课后作业
     * HeapSort ：自学 https://www.geeksforgeeks.org/heap-sort/
     * 丑数（字节跳动在半年内面试中考过）
     * 前 K 个高频元素（亚马逊在半年内面试中常考）
     * 3. 图的实现和特性
     * 思考题
     * 自己画一下有向有权图
     * 参考链接
     * 连通图个数： https://leetcode-cn.com/problems/number-of-islands/
     * 拓扑排序（Topological Sorting）： https://zhuanlan.zhihu.com/p/34871092
     * 最短路径（Shortest Path）：Dijkstra https://www.bilibili.com/video/av25829980?from=search&seid=13391343514095937158
     * 最小生成树（Minimum Spanning Tree）： https://www.bilibili.com/video/av84820276?from=search&seid=17476598104352152051
     * 4.本周学习问题反馈
     * 追赶朋友
     * 5.本周作业及下周预习
     * 本周作业
     * 简单：
     * 写一个关于 HashMap 的小总结。
     * 说明：对于不熟悉 Java 语言的同学，此项作业可选做。
     * 有效的字母异位词（亚马逊、Facebook、谷歌在半年内面试中考过）
     * 两数之和（近半年内，亚马逊考查此题达到 216 次、字节跳动 147 次、谷歌 104 次，Facebook、苹果、微软、腾讯也在近半年内面试常考）
     * N 叉树的前序遍历（亚马逊在半年内面试中考过）
     * HeapSort ：自学 https://www.geeksforgeeks.org/heap-sort/
     * 中等：
     * 字母异位词分组（亚马逊在半年内面试中常考）
     * 二叉树的中序遍历（亚马逊、字节跳动、微软在半年内面试中考过）
     * 二叉树的前序遍历（字节跳动、谷歌、腾讯在半年内面试中考过）
     * N 叉树的层序遍历（亚马逊在半年内面试中考过）
     * 丑数（字节跳动在半年内面试中考过）
     * 前 K 个高频元素（亚马逊在半年内面试中常考）
     * 下周预习
     * 预习题目：
     * 爬楼梯
     * 括号生成
     * Pow(x, n)
     * 子集
     * N 皇后
     *
     * 训练场练习（选做）
     * 学有余力的同学，可以挑战以下【训练场】模拟面试真题：
     *
     * 哈希相关：
     * 找雪花
     *
     * 树相关：
     * 安装路灯
     *
     * 二叉搜索树相关：
     * 二叉搜索树的后序遍历序列
     *
     * 堆相关：
     * 最火视频榜单
     *
     * 图相关：
     * 手游上线
     * *****************************************************************************************************************
     * 2. 实战题目解析：最小的k个数、滑动窗口最大值等问题
     *
     * 剑指 Offer 40. 最小的k个数（字节跳动在半年内面试中考过）
     * 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
     * *****************************************************************************************************************
     * 方法一:排序
     * 思路和算法
     * 对原数组从小到大排序后取出前 k 个数即可
     */
    public int[] getLeastNumbers(int[] arr, int k) {
        int[] vec = new int[k];
        Arrays.sort(arr);
        for (int i = 0; i < k; ++i) {
            vec[i] = arr[i];
        }
        return vec;
    }

    /**
     * 方法二:堆
     * 思路和算法
     * 我们用一个大根堆实时维护数组的前 k 小值。首先将前 k 个数插入大根堆中，随后从第 k+1 个数开始遍历，如果当前遍历到的数比大根堆的堆顶的数要小，就把堆顶的数弹出，再插入当前遍历到的数。
     * 最后将大根堆里的数存入数组返回即可。在下面的代码中，由于 C++ 语言中的堆（即优先队列）为大根堆，我们可以这么做。
     * 而 Python 语言中的对为小根堆，因此我们要对数组中所有的数取其相反数，才能使用小根堆维护前 k 小值。
     */
    public int[] getLeastNumbers2(int[] arr, int k) {
        int[] vec = new int[k];
        if (k == 0) { // 排除 0 的情况
            return vec;
        }
        PriorityQueue<Integer> queue = new PriorityQueue<Integer>((num1, num2) -> num2 - num1);
        for (int i = 0; i < k; ++i) {
            queue.offer(arr[i]);
        }
        for (int i = k; i < arr.length; ++i) {
            if (queue.peek() > arr[i]) {
                queue.poll();
                queue.offer(arr[i]);
            }
        }
        for (int i = 0; i < k; ++i) {
            vec[i] = queue.poll();
        }
        return vec;
    }

    /**
     * 方法三
     * 使用了快速排序的思想，用基准划分数组，得到比基准小的k-1个元素,加上基准即可得到前k小-无序
     */
    public int[] getLeastNumbers3(int[] arr, int k) {
        if (k == 0 || arr.length == 0)//如果数组为空或者是选取最小的0个数
            return new int[0];//直接返回空数组
        return quickSort(arr, 0, arr.length - 1, k - 1);//调用quickSort函数输出结果，其中k-1指的是找到下标为k-1的数
    }

    public int[] quickSort(int[] num, int left, int right, int k) {
        int point = partition(num, left, right);//调用partition函数输出划分后分界点的坐标point（point前面的数都比其小，后面的数都比其大）
        if (point == k)//如果分界点的坐标正好是k，就说明前k+1个数就是最小的k个数（即0,1,2,...，k）
            return Arrays.copyOf(num, k + 1);//调用Arrays方法输出前k+1个数
        else
            return point > k ? quickSort(num, left, point - 1, k) : quickSort(num, point + 1, right, k);
        //根据分界点的坐标继续在两个子区间中的其中一个对其进行划分，直至分界点的坐标恰好为k
    }

    public int partition(int[] num, int left, int right) {
        int l = left, r = right + 1, t = num[left];//以t=num[left]为基准对数组进行划分，即小基准大
        while (true)//用while循环完成一次对数组的划分
        {
            while (++l < right && num[l] < t) ;
            while (--r > left && num[r] > t) ;
            if (l >= r)//循环结束条件
                break;
            int temp = num[l];//num[l]比基准大，num[r]比基准小，交换两者
            num[l] = num[r];
            num[r] = temp;
        }
        num[left] = num[r];//完成对数组的一次划分后，num[r]比基准小（且是数组中比基准小的最后一个元素），将num[r]和基准交换，
        num[r] = t;
        return r;//将分界点坐标返回
    }

    /**
     * 方法四
     * 使用优先队列（大根堆）,如果规模小于k就将其入队，否则与队列中的最大值相比较，若比队列中最大值小，则将最大值出队，将其入队
     */
    public int[] getLeastNumbers4(int[] arr, int k) {
        if (k == 0 || arr.length == 0)
            return new int[0];
        Queue<Integer> queue = new PriorityQueue<>((v1, v2) -> v2 - v1);//默认小根堆，实现大根堆需要重写比较器（λ函数）
        int t = 0, res[] = new int[k];
        for (int num : arr) {
            if (queue.size() < k)//如果大根堆的规模小于k，就将数组中的数入队
                queue.add(num);
            else if (queue.peek() > num)
            //否则就取出队列中最大的数与数组中的数进行比较，如果队列中的数比较大，就将其出队，将数组中的数入队，最终即可实现队中为数组前k小
            {
                queue.poll();
                queue.add(num);
            }
        }
        for (int num : queue) {//将队列中留下的数输出到数组中，即可得到答案
            res[t++] = num;
        }
        return res;
    }

    /**
     * 方法五
     * 使用优先队列（小根堆），现将数组中的全部数入队，再依次poll出最小的数，即可得出前k小
     */
    public int[] getLeastNumbers5(int[] arr, int k) {
        if (k == 0 || arr.length == 0)
            return new int[0];
        Queue<Integer> queue = new PriorityQueue<>();//定义小根堆
        int t = 0;
        int[] res = new int[k];
        for (int num : arr)
            queue.add(num);
        while (t < k) {
            res[t++] = queue.poll().intValue();
        }
        return res;
    }

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
     * 本次解法太多了，见Week3里的解法
     *
     */

    /**
     * 剑指 Offer 49. 丑数（字节跳动在半年内面试中考过）
     * 我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
     * *****************************************************************************************************************
     * 解题思路：
     * 丑数的递推性质： 丑数只包含因子 2, 3, 5 ,因此有 “丑数 = 某较小丑数 × 某因子” （例如：10 = 5×2）。
     * 动态规划解析：
     * https://leetcode-cn.com/problems/chou-shu-lcof/solution/mian-shi-ti-49-chou-shu-dong-tai-gui-hua-qing-xi-t/
     * *****************************************************************************************************************
     * 丑数II， 合并 3 个有序数组， 清晰的推导思路 【如下】
     * https://leetcode-cn.com/problems/chou-shu-lcof/solution/chou-shu-ii-qing-xi-de-tui-dao-si-lu-by-mrsate/
     * 说白了，就是把所有丑数列出来，然后从小到大排序。而大的丑数必然是小丑数的2/3/5倍，所以有了那3个数组。每次就从那数组中取出一个最小的丑数归并到目标数组中。
     * 这题还有个关键是去重，题解用了if而不是if else，可以让重复的值比如6时,p2和p3都加一，避免了出现两个6
     */
    public int nthUglyNumber(int n) {
        int a = 0, b = 0, c = 0;
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            int n2 = dp[a] * 2, n3 = dp[b] * 3, n5 = dp[c] * 5;
            dp[i] = Math.min(Math.min(n2, n3), n5);
            if (dp[i] == n2) a++;
            if (dp[i] == n3) b++;
            if (dp[i] == n5) c++;
        }
        return dp[n - 1];
    }

    /**
     * 347. 前 K 个高频元素（亚马逊在半年内面试中常考）
     * 给定一个非空的整数数组，返回其中出现频率前 k 高的元素。
     * *****************************************************************************************************************
     * 方法一：堆
     * 链接：https://leetcode-cn.com/problems/top-k-frequent-elements/solution/qian-k-ge-gao-pin-yuan-su-by-leetcode-solution/
     */
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
        for (int num : nums) {
            occurrences.put(num, occurrences.getOrDefault(num, 0) + 1);
        }

        // int[] 的第一个元素代表数组的值，第二个元素代表了该值出现的次数
        PriorityQueue<int[]> queue = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] m, int[] n) {
                return m[1] - n[1];
            }
        });
        for (Map.Entry<Integer, Integer> entry : occurrences.entrySet()) {
            int num = entry.getKey(), count = entry.getValue();
            if (queue.size() == k) {
                if (queue.peek()[1] < count) {
                    queue.poll();
                    queue.offer(new int[]{num, count});
                }
            } else {
                queue.offer(new int[]{num, count});
            }
        }
        int[] ret = new int[k];
        for (int i = 0; i < k; ++i) {
            ret[i] = queue.poll()[0];
        }
        return ret;
    }

    /**
     * 方法二：基于快速排序
     * https://leetcode-cn.com/problems/top-k-frequent-elements/solution/qian-k-ge-gao-pin-yuan-su-by-leetcode-solution/
     */
    public int[] topKFrequent2(int[] nums, int k) {
        Map<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
        for (int num : nums) {
            occurrences.put(num, occurrences.getOrDefault(num, 0) + 1);
        }

        List<int[]> values = new ArrayList<int[]>();
        for (Map.Entry<Integer, Integer> entry : occurrences.entrySet()) {
            int num = entry.getKey(), count = entry.getValue();
            values.add(new int[]{num, count});
        }
        int[] ret = new int[k];
        qsort(values, 0, values.size() - 1, ret, 0, k);
        return ret;
    }

    public void qsort(List<int[]> values, int start, int end, int[] ret, int retIndex, int k) {
        int picked = (int) (Math.random() * (end - start + 1)) + start;
        Collections.swap(values, picked, start);

        int pivot = values.get(start)[1];
        int index = start;
        for (int i = start + 1; i <= end; i++) {
            if (values.get(i)[1] >= pivot) {
                Collections.swap(values, index + 1, i);
                index++;
            }
        }
        Collections.swap(values, start, index);

        if (k <= index - start) {
            qsort(values, start, index - 1, ret, retIndex, k);
        } else {
            for (int i = start; i <= index; i++) {
                ret[retIndex++] = values.get(i)[0];
            }
            if (k > index - start + 1) {
                qsort(values, index + 1, end, ret, retIndex, k - (index - start + 1));
            }
        }
    }

    /**
     * 3. 图的实现和特性
     * 思考题
     * 自己画一下有向有权图
     * 参考链接
     * 连通图个数： https://leetcode-cn.com/problems/number-of-islands/
     * 拓扑排序（Topological Sorting）： https://zhuanlan.zhihu.com/p/34871092
     * 最短路径（Shortest Path）：Dijkstra  https://www.bilibili.com/video/av25829980?from=search&seid=13391343514095937158
     * 最小生成树（Minimum Spanning Tree）： https://www.bilibili.com/video/av84820276?from=search&seid=17476598104352152051
     * *****************************************************************************************************************
     * 4.本周学习问题反馈
     * 追赶朋友
     * 小张和骑友小王在路上上不小心走散，小张通过定位仪找到小王的位置并且希望能快速找到小王。
     * 小张最开始在 N 点位置 (0 ≤ N ≤ 100,000)，小王显示在 K 点位置 (0 ≤ K ≤ 100,000)，
     * 小张有两种移动方式：
     * 方式一：在任意点 X，向前走一步 X + 1 或向后走一步 X - 1 需要花费一分钟
     * 方式二：在任意点 X，向前移动到 2 * X 位置需要花费一分钟
     * 假设小王就在原地等待不发生移动，那么小张找到小王最少需要多少分钟
     */
    public int findMinMinutes(int n, int k) {
        // 自己完成本题
        return 0;
    }

    public static void main(String[] args) {
        Week2 week2 = new Week2();
        // anagram : 相同字母异序词.
        String s = "washtondc";
        String t = "dcwashton";
        boolean isAnagram1 = week2.isAnagram(s, t);
        boolean isAnagram2 = week2.isAnagram2(s, t);
        System.out.println("isAnagram1 = " + isAnagram1 + "    isAnagram2 = " + isAnagram2);

        // groupAnagrams
        String[] strs = new String[]{"eat", "tea", "tan", "ate", "nat", "bat"};
        List<List<String>> list = week2.groupAnagrams(strs);
        System.out.println("groupAnagrams = " + list);
        List<List<String>> list2 = week2.groupAnagrams2(strs);
        System.out.println("groupAnagrams2 = " + list2);
        List<List<String>> list3 = week2.groupAnagrams3(strs);
        System.out.println("groupAnagrams3 = " + list3);
        List<List<String>> list4 = week2.groupAnagrams4(strs);
        System.out.println("groupAnagrams4 = " + list4);
        List<List<String>> list5 = week2.groupAnagrams5(strs);
        System.out.println("groupAnagrams5 = " + list5);
        List<List<String>> list6 = week2.groupAnagrams6(strs);
        System.out.println("groupAnagrams6 = " + list6);
        List<List<String>> list7 = week2.groupAnagrams7(strs);
        System.out.println("groupAnagrams7 = " + list7);


        //
    }
}
