学习总结

五毒神掌
过遍数，至少5遍。
不要死磕，5分钟没有思路就看题解，看懂题解自己写一遍，第二天再自己写一遍，隔一周再写一遍。
做完的题去国际站看题解，学习更优秀的代码写法。
新接触的数据结构和算法：

跳表：有序链表加多级索引
Deque：两端都可以进出的Queue
Priority Queue：可以按照元素优先级取出
Tire树：每个结点保存一个字符，共同前缀的单词共用根节点（搜索提示）
布隆过滤器Bloom Filter：一个很长的二进制向量和一系列随机映射函数。
字符串匹配算法：Rabin-Karp 算法：计算目标字符串中每个长度为M的子串的hash值(N-M+1次)，如果hash值相同，再使用朴素算法计算。计算hash的时候使用滑动窗口法，可以在O(1)时间内算出1个hash。
递归模板

递归终结条件
处理当前层逻辑
下探到下一层
清理当前层的状态
分治模板

终止条件
处理当前层逻辑（大问题如何分成子问题）
下探到下一层（解决子问题）
组装结果返回
清理当前层的状态
DP：分治+最优子结构

DFS

	visited = set()
	
	def dfs(node, visited):
		visited.add(node)
		#process current node here.
		...
		for next_node in node.children():
			if not next_node in visited:
				dfs(next node, visited)
BFS
	def BFS(graph, start, end):
		
		queue = []
		queue.append([start])
		visited.add(start)
		
		while queue:
			node = queue.popleft()
			visited.add(node)
			
			process(node)
			nodes = generate_relate_nodes(node)
			queue.push(nodes)
		
		#other processing work
二分查找
	left, right = 0, len(array)-1
	while left <= right:
		mid = (left + right)/2
		if array[mid] == target:
			#find the target!!
			break or return result
		elif array[mid]<target:
			left = mid+1
		else:
			right = mid-1
并查集
class UnionFind {
	private int count = 0;
	private int[] parent;
	
	public UnionFind(int n) {
		count = n;
		parent = new int[n];
		for (int i = 0; i < n; i++) {
			parent[i] = i;
		}
	}

	public int find(int p) {
		while (p != parent[p]) {
			parent[p] = parent[parent[p]];
			p = parent[p];
		}
		return p;
	}
	
	public void union(int p, int q) {
		int rootP = find(p);
		int rootQ = find(q);
		if (rootP == rootQ) return;
		parent[rootP] = rootQ;
		count--;
	}
}