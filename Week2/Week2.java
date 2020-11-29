package algorithm;


import java.util.*;

/**
 * @ClassName Week2
 * @Description
 * @Author Administrator
 * @Date 2020/11/24  5:09
 * @Version 1.0
 **/
public class Week2 {
    // anagram : 相同字母异序词.
    // 有效的字母异位词 说明:你可以假设字符串只包含小写字母。
    public boolean isAnagram(String s, String t) {
        if(s.length() != t.length())
            return false;
        int[] alpha = new int[26];
        for(int i = 0; i< s.length(); i++) {
            alpha[s.charAt(i) - 'a'] ++;
            alpha[t.charAt(i) - 'a'] --;
        }

        // return Arrays.stream(alpha).noneMatch(num -> num != 0); //合并下面操作
        for(int i=0;i<26;i++) {
            if (alpha[i] != 0)
                return false;
        }
        return true;
    }

    public boolean isAnagram2(String s, String t) {
        if(s.length()!=t.length()){
            return false;
        }
        char[] s1=s.toCharArray();
        char[] t2=t.toCharArray();
        Arrays.sort(s1);
        Arrays.sort(t2);
        for(int i=0;i<s1.length;i++){
            if(s1[i]!=t2[i]){
                return false;
            }
        }
        return true;
    }

    // 字母异位词分组:给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
    // *******思路:分组问题要有Key值来作为组的唯一识别标志;关键步骤是找到Key*******
    //  ["eat", "tea", "tan", "ate", "nat", "bat"]
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String,ArrayList<String>> map=new HashMap<>();
        for(String s:strs){
            char[] ch=s.toCharArray();
            Arrays.sort(ch);
            String key=String.valueOf(ch);
            if(!map.containsKey(key))
                map.put(key,new ArrayList<>());
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
    // String[] strs = new String[]{"eat", "tea", "tan", "ate", "nat", "bat"};
    // List<List<String>> list = week2.groupAnagrams(strs);
    //     for (int i = 0; i < list.size(); i++) {
    //     System.out.println(list.get(i));
    // }

    // 两数之和: 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。  nums = [2, 7, 11, 15], target = 9
    // 方法二：哈希表
    // 思路及算法
    // 注意到方法一的时间复杂度较高的原因是寻找 target - x 的时间复杂度过高。因此，我们需要一种更优秀的方法，能够快速寻找数组中是否存在目标元素。如果存在，我们需要找出它的索引。
    // 使用哈希表，可以将寻找 target - x 的时间复杂度降低到从 O(N)O(N) 降低到 O(1)O(1)。
    // 这样我们创建一个哈希表，对于每一个 x，我们首先查询哈希表中是否存在 target - x，然后将 x 插入到哈希表中，即可保证不会让 x 和自己匹配。
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hashtable = new Hashtable<Integer, Integer>();
        for (int i = 0; i < nums.length; ++i) {
            if (hashtable.containsKey(target - nums[i])) {
                return new int[]{hashtable.get(target - nums[i]), i};
            }
            hashtable.put(nums[i], i);
        }
        return new int[0];
    }
    // int[] nums = new int[]{2, 7, 11, 15};
    // int[] index = week2.twoSum(nums,22);
    //     for (int i = 0; i < index.length; i++) {
    //     System.out.println("index="+index[i]+"  num[index]="+nums[index[i]]);
    // }

    // ---------------------------------------------------------------------------------------------------------------------
    // Valid anagram
    public boolean isAnagram3(String s, String t) {
        if(s.length() != t.length()) return false;
        int [] a = new int [26];
        for(Character c : s.toCharArray()) a[c - 'a']++;
        for(Character c : t.toCharArray()) {
            if(a[c -'a'] == 0) return false;
            a[c - 'a']--;
        }
        return true;
    }

    public boolean isAnagram4(String s1, String s2) {
        int[] freq = new int[256];
        for(int i = 0; i < s1.length(); i++) freq[s1.charAt(i)]++;
        for(int i = 0; i < s2.length(); i++) if(--freq[s2.charAt(i)] < 0) return false;
        return s1.length() == s2.length();
    }

    public boolean isAnagram5(String s, String t) {
        char[] sChar = s.toCharArray();
        char[] tChar = t.toCharArray();
        Arrays.sort(sChar);
        Arrays.sort(tChar);
        return Arrays.equals(sChar, tChar);
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
        for (List<String> item: map.values()) {
            res.add(item);
        }
        return res;
    }


    // Two sum
    public int[] twoSum2(int[] nums, int target) {
        HashMap<Integer, Integer> tracker = new HashMap<Integer, Integer>();
        int len = nums.length;
        for (int i = 0; i < len; i++){
            if (tracker.containsKey(nums[i])){
                int left = tracker.get(nums[i]);
                return new int[]{left+1, i+1};
            } else {
                tracker.put(target - nums[i], i);
            }
        }
        return new int[2];
    }

    // ---------------------------------------------------------------------------------------------------------------------
    // 1. 堆和二叉堆的实现和特性
    // 参考链接
    // 维基百科：堆（Heap）
    // 堆的实现代码： https://shimo.im/docs/Lw86vJzOGOMpWZz2/
    // 2. 实战题目解析：最小的k个数、滑动窗口最大值等问题
    // 实战例题
    // 最小的 k 个数（字节跳动在半年内面试中考过）
    // 滑动窗口最大值（亚马逊在半年内面试中常考）
    // 课后作业
    // HeapSort ：自学 https://www.geeksforgeeks.org/heap-sort/
    // 丑数（字节跳动在半年内面试中考过）
    // 前 K 个高频元素（亚马逊在半年内面试中常考）
    // 3. 图的实现和特性
    // 思考题
    // 自己画一下有向有权图
    // 参考链接
    // 连通图个数： https://leetcode-cn.com/problems/number-of-islands/
    // 拓扑排序（Topological Sorting）： https://zhuanlan.zhihu.com/p/34871092
    // 最短路径（Shortest Path）：Dijkstra https://www.bilibili.com/video/av25829980?from=search&seid=13391343514095937158
    // 最小生成树（Minimum Spanning Tree）： https://www.bilibili.com/video/av84820276?from=search&seid=17476598104352152051
    // 4.本周学习问题反馈
    // 追赶朋友
    // 5.本周作业及下周预习
    // 本周作业
    // 简单：
    // 写一个关于 HashMap 的小总结。
    // 说明：对于不熟悉 Java 语言的同学，此项作业可选做。
    // 有效的字母异位词（亚马逊、Facebook、谷歌在半年内面试中考过）
    // 两数之和（近半年内，亚马逊考查此题达到 216 次、字节跳动 147 次、谷歌 104 次，Facebook、苹果、微软、腾讯也在近半年内面试常考）
    // N 叉树的前序遍历（亚马逊在半年内面试中考过）
    // HeapSort ：自学 https://www.geeksforgeeks.org/heap-sort/
    // 中等：
    // 字母异位词分组（亚马逊在半年内面试中常考）
    // 二叉树的中序遍历（亚马逊、字节跳动、微软在半年内面试中考过）
    // 二叉树的前序遍历（字节跳动、谷歌、腾讯在半年内面试中考过）
    // N 叉树的层序遍历（亚马逊在半年内面试中考过）
    // 丑数（字节跳动在半年内面试中考过）
    // 前 K 个高频元素（亚马逊在半年内面试中常考）
    // 下周预习
    // 预习题目：
    // 爬楼梯
    // 括号生成
    // Pow(x, n)
    // 子集
    // N 皇后
    // ---------------------------------------------------------------------------------------------------------------------
    // 2. 实战题目解析：最小的k个数、滑动窗口最大值等问题
    // 最小的 k 个数（字节跳动在半年内面试中考过）
    // 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
    // 方法一:排序
    // 思路和算法
    // 对原数组从小到大排序后取出前 k 个数即可
    public int[] getLeastNumbers(int[] arr, int k) {
        int[] vec = new int[k];
        Arrays.sort(arr);
        for (int i = 0; i < k; ++i) {
            vec[i] = arr[i];
        }
        return vec;
    }
    // 方法二:堆
    // 思路和算法
    // 我们用一个大根堆实时维护数组的前 k 小值。首先将前 kk 个数插入大根堆中，随后从第 k+1 个数开始遍历，如果当前遍历到的数比大根堆的堆顶的数要小，就把堆顶的数弹出，再插入当前遍历到的数。最后将大根堆里的数存入数组返回即可。在下面的代码中，由于 C++ 语言中的堆（即优先队列）为大根堆，我们可以这么做。而 Python 语言中的对为小根堆，因此我们要对数组中所有的数取其相反数，才能使用小根堆维护前 k 小值。
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
    // https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/solution/duo-chong-suan-fa-jie-jue-zui-xiao-de-kge-shu-by-e/
    // 思路：
    // 优先队列PriorityQueue是Queue接口的实现，可以对其中元素进行排序，
    // 可以放基本数据类型的包装类（如：Integer，Long等）或自定义的类
    // 对于基本数据类型的包装器类，优先队列中元素默认排列顺序是升序排列
    // 但对于自己定义的类来说，需要自己定义比较器，用λ表达式即可!---（（v1,v2）->v2-v1）
    // 以下是优先队列中常用的方法：
    // peek()//返回队首元素
    // poll()//返回队首元素，队首元素出队列
    // add()//添加元素
    // size()//返回队列元素个数
    // isEmpty()//判断队列是否为空，为空返回true,不空返回false
    // 方法三
    //使用了快速排序的思想，用基准划分数组，得到比基准小的k-1个元素,加上基准即可得到前k小-无序
    public int[] getLeastNumbers3(int[] arr, int k) {
        if(k==0||arr.length==0)//如果数组为空或者是选取最小的0个数
            return new int[0];//直接返回空数组
        return quickSort(arr,0,arr.length-1,k-1);//调用quickSort函数输出结果，其中k-1指的是找到下标为k-1的数
    }
    public int[] quickSort(int[] num,int left,int right,int k){
        int point = partition(num,left,right);//调用partition函数输出划分后分界点的坐标point（point前面的数都比其小，后面的数都比其大）
        if(point==k)//如果分界点的坐标正好是k，就说明前k+1个数就是最小的k个数（即0,1,2,...，k）
            return Arrays.copyOf(num,k+1);//调用Arrays方法输出前k+1个数
        else
            return point>k?quickSort(num,left,point-1,k):quickSort(num,point+1,right,k);
        //根据分界点的坐标继续在两个子区间中的其中一个对其进行划分，直至分界点的坐标恰好为k
    }
    public int partition(int[] num,int left,int right){
        int l=left,r=right+1,t=num[left];//以t=num[left]为基准对数组进行划分，即小基准大
        while(true)//用while循环完成一次对数组的划分
        {
            while(++l<right&&num[l]<t);
            while(--r>left&&num[r]>t);
            if(l>=r)//循环结束条件
                break;
            int temp=num[l];//num[l]比基准大，num[r]比基准小，交换两者
            num[l]=num[r];
            num[r]=temp;
        }
        num[left]=num[r];//完成对数组的一次划分后，num[r]比基准小（且是数组中比基准小的最后一个元素），将num[r]和基准交换，
        num[r]=t;
        return r;//将分界点坐标返回
    }
    // 方法四
    // 使用优先队列（大根堆）,如果规模小于k就将其入队，否则与队列中的最大值相比较，若比队列中最大值小，则将最大值出队，将其入队
    public int[] getLeastNumbers4(int[] arr, int k) {
        if(k==0||arr.length==0)
            return new int[0];
        Queue<Integer> queue=new PriorityQueue<>((v1,v2)->v2 - v1);//默认小根堆，实现大根堆需要重写比较器（λ函数）
        int t=0,res[]=new int[k];
        for(int num:arr){
            if(queue.size()<k)//如果大根堆的规模小于k，就将数组中的数入队
                queue.add(num);
            else if(queue.peek()>num)
            //否则就取出队列中最大的数与数组中的数进行比较，如果队列中的数比较大，就将其出队，将数组中的数入队，最终即可实现队中为数组前k小
            {
                queue.poll();
                queue.add(num);
            }
        }
        for(int num:queue){//将队列中留下的数输出到数组中，即可得到答案
            res[t++]=num;
        }
        return res;
    }
    // 方法五
    // 使用优先队列（小根堆），现将数组中的全部数入队，再依次poll出最小的数，即可得出前k小
    public int[] getLeastNumbers5(int[] arr, int k) {
        if(k==0||arr.length==0)
            return new int[0];
        Queue<Integer> queue=new PriorityQueue<>();//定义小根堆
        int t=0;
        int[] res =new int[k];
        for(int num:arr)
            queue.add(num);
        while(t<k){
            res[t++]=queue.poll().intValue();
        }
        return res;
    }

    // 4.本周学习问题反馈
    // 追赶朋友
    // 小张和骑友小王在路上上不小心走散，小张通过定位仪找到小王的位置并且希望能快速找到小王。
    // 小张最开始在 N 点位置 (0 ≤ N ≤ 100,000)，小王显示在 K 点位置 (0 ≤ K ≤ 100,000)，
    // 小张有两种移动方式：
    // 方式一：在任意点 X，向前走一步 X + 1 或向后走一步 X - 1 需要花费一分钟
    // 方式二：在任意点 X，向前移动到 2 * X 位置需要花费一分钟
    // 假设小王就在原地等待不发生移动，那么小张找到小王最少需要多少分钟
    public int findMinMinutes(int n, int k) {
        return 0;
    }

    // 丑数（字节跳动在半年内面试中考过）:我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
    // 解题思路：
    // 丑数的递推性质： 丑数只包含因子 2, 3, 5 ,因此有 “丑数 = 某较小丑数 × 某因子” （例如：10 = 5×2）。
    // 动态规划解析：https://leetcode-cn.com/problems/chou-shu-lcof/solution/mian-shi-ti-49-chou-shu-dong-tai-gui-hua-qing-xi-t/
    // 丑数II， 合并 3 个有序数组， 清晰的推导思路   https://leetcode-cn.com/problems/chou-shu-lcof/solution/chou-shu-ii-qing-xi-de-tui-dao-si-lu-by-mrsate/
    // 说白了，就是把所有丑数列出来，然后从小到大排序。而大的丑数必然是小丑数的2/3/5倍，所以有了那3个数组。每次就从那数组中取出一个最小的丑数归并到目标数组中。
    public int nthUglyNumber(int n) {
        int a = 0, b = 0, c = 0;
        int[] dp = new int[n];
        dp[0] = 1;
        for(int i = 1; i < n; i++) {
            int n2 = dp[a] * 2, n3 = dp[b] * 3, n5 = dp[c] * 5;
            dp[i] = Math.min(Math.min(n2, n3), n5);
            if(dp[i] == n2) a++;
            if(dp[i] == n3) b++;
            if(dp[i] == n5) c++;
        }
        return dp[n - 1];
    }
    public int nthUglyNumber2(int n) {
        // ....
        // dp[i] = min(min(dp[p2] * 2, dp[p3] * 3), dp[p5] * 5);
        // if (dp[i] == dp[p2] * 2)
        //     p2++;
        // if (dp[i] == dp[p3] * 3)
        //     p3++;
        // if (dp[i] == dp[p5] * 5)
        //     p5++;
        // ......
        // 合并过程中重复解的处理:
        // nums2, nums3, nums5 中是存在重复的解的， 例如 nums2[2] == 3*2, nums3[1] == 2*3 都计算出了 6 这个结果，
        // 所以在合并 3 个有序数组的过程中， 还需要跳过相同的结果， 这也就是为什么在比较的时候， 需要使用 3 个并列的 if... if... if... 而不是 if... else if... else 这种结构的原因。
        // 当比较到元素 6 时， if (dp[i] == dp[p2] * 2)...if (dp[i] == dp[p3] * 3)... 可以同时指向 nums2, nums3 中 元素 6 的下一个元素。
        // vector<int> dp(n, 0);
        // dp[0] = 1;
        // int p2 = 0, p3 = 0, p5 = 0;
        // for (int i = 1; i < n; i++) {
        //     dp[i] = min(min(dp[p2] * 2, dp[p3] * 3), dp[p5] * 5);
        //     if (dp[i] == dp[p2] * 2)
        //         p2++;
        //     if (dp[i] == dp[p3] * 3)
        //         p3++;
        //     if (dp[i] == dp[p5] * 5)
        //         p5++;
        // }
        // return dp[n - 1];
        return 1;
    }






}
