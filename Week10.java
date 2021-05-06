package algorithm;

import java.util.*;

/**
 * @ClassName Week10
 * @Description TODO
 * @Author Administrator
 * @Date 2021/1/18  9:38
 * @Version 1.0
 **/
public class Week10 {

    /**
     * 剑指 Offer 32 - III. 从上到下打印二叉树 III
     * 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
     * *****************************************************************************************************************
     * 例如:
     * 给定二叉树:[3,9,20,null,null,15,7],
     *
     *     3
     *    / \
     *   9  20
     *     /  \
     *    15   7
     * 返回其层次遍历结果：
     *
     * [
     *   [3],
     *   [20,9],
     *   [15,7]
     * ]
     * *****************************************************************************************************************
     * 解题思路：
     * 面试题32 - I. 从上到下打印二叉树 主要考察 树的按层打印 ；
     * 面试题32 - II. 从上到下打印二叉树 II 额外要求 每一层打印到一行 ；
     * 本题额外要求 打印顺序交替变化（建议按顺序做此三道题）。
     *
     * 方法一：层序遍历 + 双端队列
     * 利用双端队列的两端皆可添加元素的特性，设打印列表（双端队列） tmp ，并规定：
     * 奇数层 则添加至 tmp 尾部 ，
     * 偶数层 则添加至 tmp 头部 。
     * 算法流程：
     * 特例处理： 当树的根节点为空，则直接返回空列表 [] ；
     * 初始化： 打印结果空列表 res ，包含根节点的双端队列 deque ；
     * BFS 循环： 当 deque 为空时跳出；
     * 新建列表 tmp ，用于临时存储当前层打印结果；
     * 当前层打印循环： 循环次数为当前层节点数（即 deque 长度）；
     * 出队： 队首元素出队，记为 node；
     * 打印： 若为奇数层，将 node.val 添加至 tmp 尾部；否则，添加至 tmp 头部；
     * 添加子节点： 若 node 的左（右）子节点不为空，则加入 deque ；
     * 将当前层结果 tmp 转化为 list 并添加入 res ；
     * 返回值： 返回打印结果列表 res 即可；
     * 链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/solution/mian-shi-ti-32-iii-cong-shang-dao-xia-da-yin-er--3/
     * 复杂度分析：
     * 时间复杂度 O(N) ： N 为二叉树的节点数量，即 BFS 需循环 N 次，占用 O(N) ；双端队列的队首和队尾的添加和删除操作的时间复杂度均为 O(1) 。
     * 空间复杂度 O(N) ： 最差情况下，即当树为满二叉树时，最多有 N/2 个树节点 同时 在 deque 中，使用 O(N) 大小的额外空间。
     * 代码：
     * Python 中使用 collections 中的双端队列 deque() ，其 popleft() 方法可达到 O(1) 时间复杂度；列表 list 的 pop(0) 方法时间复杂度为 O(N) 。
     * Java 中将链表 LinkedList 作为双端队列使用。
     *
     * 作者：jyd
     * 链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/solution/mian-shi-ti-32-iii-cong-shang-dao-xia-da-yin-er--3/
     * 来源：力扣（LeetCode）
     * 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
     */
    public List<List<Integer>> levelOrder31(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if (root != null) queue.add(root);
        while (!queue.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                if (res.size() % 2 == 0) tmp.addLast(node.val); // 偶数层 -> 队列头部
                else tmp.addFirst(node.val); // 奇数层 -> 队列尾部
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            res.add(tmp);
        }
        return res;
    }
    /**
     * 方法二：层序遍历 + 双端队列（奇偶层逻辑分离）
     * 方法一代码简短、容易实现；但需要判断每个节点的所在层奇偶性，即冗余了 NN 次判断。
     * 通过将奇偶层逻辑拆分，可以消除冗余的判断。
     * 算法流程：
     * 与方法一对比，仅 BFS 循环不同。
     *
     * BFS 循环： 循环打印奇 / 偶数层，当 deque 为空时跳出；
     * 1.打印奇数层： 从左向右 打印，先左后右 加入下层节点；
     * 2.若 deque 为空，说明向下无偶数层，则跳出；
     * 3.打印偶数层： 从右向左 打印，先右后左 加入下层节点；
     * 复杂度分析：
     * 时间复杂度 O(N) ： 同方法一。
     * 空间复杂度 O(N) ： 同方法一。
     * 代码：
     *
     * 作者：jyd
     * 链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/solution/mian-shi-ti-32-iii-cong-shang-dao-xia-da-yin-er--3/
     * 来源：力扣（LeetCode）
     * 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
     */
    public List<List<Integer>> levelOrder32(TreeNode root) {
        Deque<TreeNode> deque = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if (root != null) deque.add(root);
        while (!deque.isEmpty()) {
            // 打印奇数层
            List<Integer> tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                // 从左向右打印
                TreeNode node = deque.removeFirst();
                tmp.add(node.val);
                // 先左后右加入下层节点
                if (node.left != null) deque.addLast(node.left);
                if (node.right != null) deque.addLast(node.right);
            }
            res.add(tmp);
            if (deque.isEmpty()) break; // 若为空则提前跳出
            // 打印偶数层
            tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                // 从右向左打印
                TreeNode node = deque.removeLast();
                tmp.add(node.val);
                // 先右后左加入下层节点
                if (node.right != null) deque.addFirst(node.right);
                if (node.left != null) deque.addFirst(node.left);
            }
            res.add(tmp);
        }
        return res;
    }

    /**
     * 方法三：层序遍历 + 倒序
     * 此方法的优点是只用列表即可，无需其他数据结构。
     * 偶数层倒序： 若 res 的长度为 奇数 ，说明当前是偶数层，则对 tmp 执行 倒序 操作。
     * 复杂度分析：
     * 时间复杂度 O(N) ： N 为二叉树的节点数量，即 BFS 需循环 N 次，占用 O(N) 。共完成 少于 N 个节点的倒序操作，占用 O(N) 。
     * 空间复杂度 O(N) ： 最差情况下，即当树为满二叉树时，最多有 N/2 个树节点同时在 queue 中，使用 O(N) 大小的额外空间。
     *
     * 作者：jyd
     * 链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/solution/mian-shi-ti-32-iii-cong-shang-dao-xia-da-yin-er--3/
     * 来源：力扣（LeetCode）
     * 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
     * *****************************************************************************************************************
     * 大佬确定奇偶的方法好聪明啊 !!!
     */
    public List<List<Integer>> levelOrder33(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if (root != null) queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();
            for (int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                tmp.add(node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            if (res.size() % 2 == 1) Collections.reverse(tmp);
            res.add(tmp);
        }
        return res;
    }

    /**
     * 剑指 Offer 32 - II. 从上到下打印二叉树 II
     * 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
     *
     *
     * 例如:
     * 给定二叉树: [3,9,20,null,null,15,7],
     *
     *     3
     *    / \
     *   9  20
     *     /  \
     *    15   7
     * 返回其层次遍历结果：
     *
     * [
     *   [3],
     *   [9,20],
     *   [15,7]
     * ]
     *
     * *****************************************************************************************************************
     * 面试题32 - II. 从上到下打印二叉树 II（层序遍历 BFS，清晰图解）
     * 解题思路：
     * 建议先做 面试题32 - I. 从上到下打印二叉树 再做此题，两题仅有微小区别，即本题需将 每一层打印到一行 。
     *
     * I. 按层打印： 题目要求的二叉树的 从上至下 打印（即按层打印），又称为二叉树的 广度优先搜索（BFS）。BFS 通常借助 队列 的先入先出特性来实现。
     *
     * II. 每层打印到一行： 将本层全部节点打印到一行，并将下一层全部节点加入队列，以此类推，即可分为多行打印。
     *
     *
     *
     * 算法流程：
     * 特例处理： 当根节点为空，则返回空列表 [] ；
     * 初始化： 打印结果列表 res = [] ，包含根节点的队列 queue = [root] ；
     * BFS 循环： 当队列 queue 为空时跳出；
     * 新建一个临时列表 tmp ，用于存储当前层打印结果；
     * 当前层打印循环： 循环次数为当前层节点数（即队列 queue 长度）；
     * 出队： 队首元素出队，记为 node；
     * 打印： 将 node.val 添加至 tmp 尾部；
     * 添加子节点： 若 node 的左（右）子节点不为空，则将左（右）子节点加入队列 queue ；
     * 将当前层结果 tmp 添加入 res 。
     * 返回值： 返回打印结果列表 res 即可。
     *
     *
     * 作者：jyd
     * 链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/solution/mian-shi-ti-32-ii-cong-shang-dao-xia-da-yin-er-c-5/
     * 来源：力扣（LeetCode）
     * 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
     */
    public List<List<Integer>> levelOrder21(TreeNode root) {


        return null;
    }



}
