package algorithm;

import javax.swing.tree.TreeNode;
import java.util.*;

/**
 * *********************************************************************************************************************
 * 236.Lowest Common Ancestor of a Binary Tree (LCA)
 * Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
 *
 * According to the definition of LCA on Wikipedia:
 * “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
 * *********************************************************************************************************************
 * Solution
 * First the given nodes p and q are to be searched in a binary tree and then their lowest common ancestor is to be found.
 * We can resort to a normal tree traversal to search for the two nodes.
 * Once we reach the desired nodes p and q, we can backtrack and find the lowest common ancestor.
 * *********************************************************************************************************************
 * Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
 * Output: 3
 * Explanation: The LCA of nodes 5 and 1 is 3.
 * <br>
 * Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
 * Output: 5
 * Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
 * <br>
 * Input: root = [1,2], p = 1, q = 2
 * Output: 1
 **/
public class Solution {
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     *
     * https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/solution/
     */
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    private TreeNode ans;

    public Solution() {
        // Variable to store LCA node.
        this.ans = null;
    }

    private boolean recurseTree(TreeNode currentNode, TreeNode p, TreeNode q) {

        // If reached the end of a branch, return false.
        if (currentNode == null) {
            return false;
        }

        // Left Recursion. If left recursion returns true, set left = 1 else 0
        int left = this.recurseTree(currentNode.left, p, q) ? 1 : 0;

        // Right Recursion
        int right = this.recurseTree(currentNode.right, p, q) ? 1 : 0;

        // If the current node is one of p or q
        int mid = (currentNode == p || currentNode == q) ? 1 : 0;


        // If any two of the flags left, right or mid become True
        if (mid + left + right >= 2) {
            this.ans = currentNode;
        }

        // Return true if any one of the three bool values is True.
        return (mid + left + right > 0);
    }

    /**
     * Approach 1: Recursive Approach
     *
     * Intuition
     *
     * The approach is pretty intuitive. Traverse the tree in a depth first manner.
     * The moment you encounter either of the nodes p or q, return some boolean flag.
     * The flag helps to determine if we found the required nodes in any of the paths.
     * The least common ancestor would then be the node for which both the subtree recursions return a True flag.
     * It can also be the node which itself is one of p or q and for which one of the subtree recursions returns a True flag.
     *
     * Let us look at the formal algorithm based on this idea.
     *
     * Algorithm
     *
     * 1.Start traversing the tree from the root node.
     * 2.If the current node itself is one of p or q, we would mark a variable mid as True and continue the search for the other node in the left and right branches.
     * 3.If either of the left or the right branch returns True, this means one of the two nodes was found below.
     * 4.If at any point in the traversal, any two of the three flags left, right or mid become True, this means we have found the lowest common ancestor for the nodes p and q.
     *
     * Let us look at a sample tree and we search for the lowest common ancestor of two nodes 9 and 11 in the tree.
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // Traverse the tree
        this.recurseTree(root, p, q);
        return this.ans;
    }

    /**
     * Approach 2: Iterative using parent pointers
     *
     * Intuition
     *
     * If we have parent pointers for each node we can traverse back from p and q to get their ancestors. The first common node we get during this traversal would be the LCA node. We can save the parent pointers in a dictionary as we traverse the tree.
     *
     * Algorithm
     *
     * 1.Start from the root node and traverse the tree.
     * 2.Until we find p and q both, keep storing the parent pointers in a dictionary.
     * 3.Once we have found both p and q, we get all the ancestors for p using the parent dictionary and add to a set called ancestors.
     * 4.Similarly, we traverse through ancestors for node q. If the ancestor is present in the ancestors set for p, this means this is the first ancestor common between p and q (while traversing upwards) and hence this is the LCA node.
     */
    public TreeNode lowestCommonAncestor4(TreeNode root, TreeNode p, TreeNode q) {

        // Stack for tree traversal
        Deque<TreeNode> stack = new ArrayDeque<>();

        // HashMap for parent pointers
        Map<TreeNode, TreeNode> parent = new HashMap<>();

        parent.put(root, null);
        stack.push(root);

        // Iterate until we find both the nodes p and q
        while (!parent.containsKey(p) || !parent.containsKey(q)) {

            TreeNode node = stack.pop();

            // While traversing the tree, keep saving the parent pointers.
            if (node.left != null) {
                parent.put(node.left, node);
                stack.push(node.left);
            }
            if (node.right != null) {
                parent.put(node.right, node);
                stack.push(node.right);
            }
        }

        // Ancestors set() for node p.
        Set<TreeNode> ancestors = new HashSet<>();

        // Process all ancestors for node p using parent pointers.
        while (p != null) {
            ancestors.add(p);
            p = parent.get(p);
        }

        // The first ancestor of q which appears in
        // p's ancestor set() is their lowest common ancestor.
        while (!ancestors.contains(q))
            q = parent.get(q);
        return q;
    }



    //recursion that goes left and right and checks if node is either
    //go through and check if children are equal to either p or q
    //if right and left are true or current node is true then current node is the one
    TreeNode lca = null;
    public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode p, TreeNode q) {
        ancestorHelp(root,p,q);
        return lca;
    }
    public boolean ancestorHelp(TreeNode root, TreeNode p, TreeNode q){
        if(root == null){return false;}
        boolean found = false;
        if(root == p || root == q){
            found = true;
        }
        if(ancestorHelp(root.left, p, q)){
            if(found){
                if(lca == null){lca = root;}
                return true;
            }
            found = true;
        }
        if(lca != null){return true;}
        if(ancestorHelp(root.right, p, q)){
            if(found){
                if(lca == null){lca = root;}
                return true;
            }
            found = true;
        }
        return found;
    }

    /**
     * 105. 从前序与中序遍历序列构造二叉树
     * 105. Construct Binary Tree from Preorder and Inorder Traversal
     * Given preorder and inorder traversal of a tree, construct the binary tree.
     *
     * Note:
     * You may assume that duplicates do not exist in the tree.
     *
     * For example, given
     *
     * preorder = [3,9,20,15,7]
     * inorder = [9,3,15,20,7]
     */
    private int in = 0;
    private int pre = 0;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return build(preorder, inorder, Integer.MIN_VALUE);
    }

    private TreeNode build(int[] preorder, int[] inorder, int stop) {
        if (pre >= preorder.length) return null;
        if (inorder[in] == stop) {
            in++;
            return null;
        }
        TreeNode node = new TreeNode(preorder[pre++]);
        node.left = build(preorder, inorder, node.val);
        node.right = build(preorder, inorder, stop);
        return node;
    }
}
