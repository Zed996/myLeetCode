package test

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	_ "strconv"
)

/*
101. 对称二叉树
双指针，一个左另一个右，判断val
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.7 MB, 在所有 Go 提交中击败了65.83%的用户
通过测试用例：198 / 198
*/
func isSymmetric(root *TreeNode) bool {
	var check func(i, j *TreeNode) bool
	check = func(i, j *TreeNode) bool {
		if i == nil && j == nil {
			return true
		}
		if i != nil && j != nil {
			if i.Val != j.Val {
				return false
			}
			return check(i.Left, j.Right) && check(i.Right, j.Left)
		}
		return false
	}
	return check(root, root)
}

/*
102. 二叉树的层序遍历
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.6 MB, 在所有 Go 提交中击败了99.35%的用户
通过测试用例：34 / 34
*/
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	stack := []*TreeNode{root}
	var res [][]int
	for i := 0; len(stack) > 0; i++ {
		arr := make([]int, len(stack))
		var tmp []*TreeNode
		for j := 0; j < len(stack); j++ {
			n := stack[j]
			arr[j] = n.Val
			if n.Left != nil {
				tmp = append(tmp, n.Left)
			}
			if n.Right != nil {
				tmp = append(tmp, n.Right)
			}
		}
		res = append(res, arr)
		stack = tmp
	}
	return res
}

/*
103. 二叉树的锯齿形层序遍历
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了94.28%的用户
通过测试用例：33 / 33
*/
func zigzagLevelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	stack := []*TreeNode{root}
	var res [][]int
	for i := 0; len(stack) > 0; i++ {
		arr := make([]int, len(stack))
		var tmp []*TreeNode

		for j := 0; j < len(stack); j++ {
			n := stack[j]
			if i%2 == 0 {
				arr[j] = n.Val
			} else {
				arr[len(stack)-j-1] = n.Val
			}
			if n.Left != nil {
				tmp = append(tmp, n.Left)
			}
			if n.Right != nil {
				tmp = append(tmp, n.Right)
			}
		}

		res = append(res, arr)
		stack = tmp
	}
	return res
}

/*
104. 二叉树的最大深度
执行用时：4 ms, 在所有 Go 提交中击败了85.62%的用户
内存消耗：4.1 MB, 在所有 Go 提交中击败了11.44%的用户
通过测试用例：39 / 39
*/
func maxDepth(root *TreeNode) int {
	max := 0
	var recursion func(n *TreeNode, level int)
	recursion = func(n *TreeNode, level int) {
		if n == nil {
			return
		}
		level++
		if level > max {
			max = level
		}
		recursion(n.Left, level)
		recursion(n.Right, level)
	}
	recursion(root, 0)
	return max
}

/*
105. 从前序与中序遍历序列构造二叉树
执行用时：4 ms, 在所有 Go 提交中击败了92.84%的用户
内存消耗：4.1 MB, 在所有 Go 提交中击败了28.14%的用户
通过测试用例：203 / 203
*/
func buildTree105(preorder []int, inorder []int) *TreeNode {
	var build func(preArr, inArr []int) *TreeNode
	build = func(preArr, inArr []int) *TreeNode {
		if len(preArr) == 0 {
			return nil
		}
		//preorder[0]必定是根节点
		res := &TreeNode{
			Val:   preArr[0],
			Left:  nil,
			Right: nil,
		}
		//记录inArr中根节点的index
		in := 0
		for i := 0; i < len(preArr); i++ {
			if preArr[0] == inArr[i] {
				in = i
				break
			}
		}
		res.Left = build(preArr[1:in+1], inArr[0:in])
		res.Right = build(preArr[in+1:], inArr[in+1:])
		return res
	}
	return build(preorder, inorder)
}

/*
106. 从中序与后序遍历序列构造二叉树
执行用时：4 ms, 在所有 Go 提交中击败了87.99%的用户
内存消耗：4 MB, 在所有 Go 提交中击败了74.76%的用户
通过测试用例：202 / 202
*/
func buildTree106(inorder []int, postorder []int) *TreeNode {
	var build func(inArr, postArr []int) *TreeNode
	build = func(inArr, postArr []int) *TreeNode {
		if len(postArr) == 0 {
			return nil
		}
		res := &TreeNode{
			Val:   postArr[len(postArr)-1],
			Left:  nil,
			Right: nil,
		}
		in := 0
		for i := 0; i < len(postArr); i++ {
			if postArr[len(postArr)-1] == inArr[i] {
				in = i
				break
			}
		}
		res.Left = build(inArr[0:in], postArr[0:in])
		res.Right = build(inArr[in+1:], postArr[in:len(postArr)-1])
		return res
	}
	return build(inorder, postorder)
}

/*
107. 二叉树的层序遍历 II
执行用时：4 ms, 在所有 Go 提交中击败了9.56%的用户
内存消耗：6.5 MB, 在所有 Go 提交中击败了15.63%的用户
通过测试用例：34 / 34
*/
func levelOrderBottom(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	stack := []*TreeNode{root}
	var res [][]int
	for i := 0; len(stack) > 0; i++ {
		var tmp []*TreeNode
		var arr []int
		for j := 0; j < len(stack); j++ {
			n := stack[j]
			arr = append(arr, n.Val)
			if n.Left != nil {
				tmp = append(tmp, n.Left)
			}
			if n.Right != nil {
				tmp = append(tmp, n.Right)
			}
		}
		stack = tmp
		res = append([][]int{arr}, res...)
	}
	return res
}

/*
108. 将有序数组转换为二叉搜索树
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：3.3 MB, 在所有 Go 提交中击败了99.92%的用户
通过测试用例：31 / 31
*/
func sortedArrayToBST(nums []int) *TreeNode {
	var build func(arr []int) *TreeNode
	build = func(arr []int) *TreeNode {
		if len(arr) == 0 {
			return nil
		}
		mid := len(arr) / 2
		return &TreeNode{
			Val:   arr[mid],
			Left:  build(arr[:mid]),
			Right: build(arr[mid+1:]),
		}
	}
	return build(nums)
}

/*
109. 有序链表转换二叉搜索树
执行用时：4 ms, 在所有 Go 提交中击败了92.77%的用户
内存消耗：6.5 MB, 在所有 Go 提交中击败了10.33%的用户
通过测试用例：32 / 32
*/
func sortedListToBST(head *ListNode) *TreeNode {
	var arr []int
	for head != nil {
		arr = append(arr, head.Val)
		head = head.Next
	}
	var build func(arr []int) *TreeNode
	build = func(arr []int) *TreeNode {
		if len(arr) == 0 {
			return nil
		}
		mid := len(arr) / 2
		return &TreeNode{
			Val:   arr[mid],
			Left:  build(arr[:mid]),
			Right: build(arr[mid+1:]),
		}
	}
	return build(arr)
}

/*
110. 平衡二叉树
执行用时：4 ms, 在所有 Go 提交中击败了94.88%的用户
内存消耗：5.6 MB, 在所有 Go 提交中击败了30.03%的用户
通过测试用例：228 / 228
*/
func isBalanced(root *TreeNode) bool {
	var height func(node *TreeNode) int
	height = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		l := height(node.Left)
		r := height(node.Right)
		if l == -1 || r == -1 || math.Abs(float64(l-r)) > 1.0 {
			return -1
		}
		if l > r {
			return l + 1
		}
		return r + 1
	}
	return height(root) != -1
}

/*
111. 二叉树的最小深度
执行用时：172 ms, 在所有 Go 提交中击败了28.04%的用户
内存消耗：22.1 MB, 在所有 Go 提交中击败了8.06%的用户
通过测试用例：52 / 52
*/
func minDepth(root *TreeNode) int {
	var min func(x, y int) int
	min = func(x, y int) int {
		if x < y {
			return x
		}
		return y
	}
	var height func(node *TreeNode) int
	height = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		if node.Left == nil && node.Right == nil {
			return 1
		}
		m := 100001
		if node.Left != nil {
			m = min(height(node.Left)+1, m)
		}
		if node.Right != nil {
			m = min(height(node.Right)+1, m)
		}
		return m
	}
	return height(root)
}

/*
112. 路径总和
执行用时：4 ms, 在所有 Go 提交中击败了89.25%的用户
内存消耗：4.4 MB, 在所有 Go 提交中击败了17.69%的用户
通过测试用例：117 / 117
*/
func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	if targetSum == root.Val && root.Left == nil && root.Right == nil {
		return true
	}
	return hasPathSum(root.Left, targetSum-root.Val) || hasPathSum(root.Right, targetSum-root.Val)
}

/*
113. 路径总和 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：4.3 MB, 在所有 Go 提交中击败了89.24%的用户
通过测试用例：115 / 115
*/
func PathSum(root *TreeNode, targetSum int) [][]int {
	if root == nil {
		return [][]int{}
	}
	var res [][]int
	var arr []int
	var sum func(node *TreeNode, target int)
	sum = func(node *TreeNode, target int) {
		if node == nil {
			return
		}
		arr = append(arr, node.Val)
		if target == node.Val && node.Left == nil && node.Right == nil {
			res = append(res, append([]int{}, arr...))
			arr = arr[:len(arr)-1]
			return
		}
		sum(node.Left, target-node.Val)
		sum(node.Right, target-node.Val)
		arr = arr[:len(arr)-1]
	}
	sum(root, targetSum)
	return res
}

/*
114. 二叉树展开为链表
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.8 MB, 在所有 Go 提交中击败了69.38%的用户
通过测试用例：225 / 225
*/
func Flatten(root *TreeNode) {
	if root == nil {
		return
	}
	for root.Left != nil || root.Right != nil {
		if root.Left != nil {
			tmp := root.Left
			for tmp.Right != nil {
				tmp = tmp.Right
			}
			r := root.Right
			tmp.Right = r
			root.Right = root.Left
			root.Left = nil
		}
		root = root.Right
	}
}

/*
115. 不同的子序列
动态规划，dp[i][j] 表示在 s[i:] 的子序列中 t[j:] 出现的个数
s[i]=t[j]:dp[i][j]=dp[i+1][j+1]+dp[i+1][j]
s[i]!=t[j]:dp[i][j]=dp[i+1][j]
执行用时：4 ms, 在所有 Go 提交中击败了85.87%的用户
内存消耗：14.3 MB, 在所有 Go 提交中击败了39.53%的用户
通过测试用例：64 / 64
*/
func numDistinct(s string, t string) int {
	ls, lt := len(s), len(t)
	if ls < lt {
		return 0
	}
	dp := make([][]int, ls+1)
	for i := range dp {
		dp[i] = make([]int, lt+1)
		dp[i][lt] = 1
	}
	for i := ls - 1; i >= 0; i-- {
		for j := lt - 1; j >= 0; j-- {
			if s[i] == t[j] {
				dp[i][j] = dp[i+1][j+1] + dp[i+1][j]
			} else {
				dp[i][j] = dp[i+1][j]
			}
		}
	}
	return dp[0][0]
}

/*type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}*/

/*
116. 填充每个节点的下一个右侧节点指针
执行用时：4 ms, 在所有 Go 提交中击败了90.05%的用户
内存消耗：6.2 MB, 在所有 Go 提交中击败了80.30%的用户
通过测试用例：59 / 59
*/
/*func connect(root *Node) *Node {
	var recursion func(n *Node)
	recursion = func(n *Node) {
		if n == nil || n.Left == nil {
			return
		}
		n.Left.Next = n.Right
		if n.Next != nil {
			n.Right.Next = n.Next.Left
		} else {
			n.Right.Next = nil
		}
		recursion(n.Left)
		recursion(n.Right)
	}
	recursion(root)
	return root
}*/

/*
117. 填充每个节点的下一个右侧节点指针 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：6 MB, 在所有 Go 提交中击败了99.33%的用户
通过测试用例：55 / 55
*/
/*func connect117(root *Node) *Node {
	start := root
	for start != nil {
		var next, last *Node
		//handle每次将current的前一个的next指向current
		handle := func(cur *Node) {
			if cur == nil {
				return
			}
			if next == nil {
				next = cur
			}
			if last != nil {
				last.Next = cur
			}
			last = cur
		}
		//处理下一层
		for tmp := start; tmp != nil; tmp = tmp.Next {
			handle(tmp.Left)
			handle(tmp.Right)
		}
		start = next
	}
	return root
}*/

/*
118. 杨辉三角
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了86.05%的用户
通过测试用例：14 / 14
*/
func generate(numRows int) [][]int {
	res := make([][]int, numRows)
	for i := 0; i < numRows; i++ {
		res[i] = make([]int, i+1)
		res[i][0] = 1
		res[i][i] = 1
		for j := 1; j < i; j++ {
			res[i][j] = res[i-1][j-1] + res[i-1][j]
		}
	}
	return res
}

/*
119. 杨辉三角 II
组合数公式
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了55.00%的用户
通过测试用例：34 / 34
*/
func getRow(rowIndex int) []int {
	res := make([]int, rowIndex+1)
	res[0] = 1
	for i := 1; i <= rowIndex; i++ {
		res[i] = res[i-1] * (rowIndex - i + 1) / i
	}
	return res
}

/*
120. 三角形最小路径和
执行用时：4 ms, 在所有 Go 提交中击败了92.79%的用户
内存消耗：3.1 MB, 在所有 Go 提交中击败了68.25%的用户
通过测试用例：44 / 44
*/
func minimumTotal(triangle [][]int) int {
	m := make([]int, len(triangle)+1)
	//自底而上
	for i := len(triangle) - 1; i >= 0; i-- {
		for j := 0; j <= i; j++ {
			m[j] = int(math.Min(float64(m[j]), float64(m[j+1]))) + triangle[i][j]
		}
	}
	return m[0]
}

/*
121. 买卖股票的最佳时机
执行用时：96 ms, 在所有 Go 提交中击败了90.64%的用户
内存消耗：7.7 MB, 在所有 Go 提交中击败了75.98%的用户
通过测试用例：211 / 211
*/
func maxProfit121(prices []int) int {
	min := 10001
	res := 0
	for i := 0; i < len(prices); i++ {
		if prices[i] < min {
			min = prices[i]
		} else if prices[i]-min > res {
			res = prices[i] - min
		}
	}
	return res
}

/*
122. 买卖股票的最佳时机 II
遍历整个股票交易日价格列表 price，策略是所有上涨交易日都买卖（赚到所有利润），所有下降交易日都不买卖（永不亏钱）。
执行用时：4 ms, 在所有 Go 提交中击败了89.78%的用户
内存消耗：2.9 MB, 在所有 Go 提交中击败了70.66%的用户
通过测试用例：200 / 200
*/
func maxProfit122(prices []int) int {
	total := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] > prices[i-1] {
			total += prices[i] - prices[i-1]
		}
	}
	return total
}

/*
123. 买卖股票的最佳时机 III
动态规划
buy1=max{buy1,−prices[i]}
sell1=max{sell1,buy1+prices[i]}
buy2=max{buy2,sell1−prices[i]}
sell2=max{sell2,buy2+prices[i]}
执行用时：116 ms, 在所有 Go 提交中击败了41.77%的用户
内存消耗：8.6 MB, 在所有 Go 提交中击败了60.97%的用户
通过测试用例：214 / 214
*/
func maxProfit123(prices []int) int {
	buy1, sell1, buy2, sell2 := -prices[0], 0, -prices[0], 0
	for i := 1; i < len(prices); i++ {
		buy1 = max(buy1, -prices[i])
		sell1 = max(sell1, buy1+prices[i])
		buy2 = max(buy2, sell1-prices[i])
		sell2 = max(sell2, buy2+prices[i])
	}
	return sell2
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

/*
124. 二叉树中的最大路径和
执行用时：16 ms, 在所有 Go 提交中击败了73.64%的用户
内存消耗：7.4 MB, 在所有 Go 提交中击败了19.63%的用户
通过测试用例：94 / 94
*/
func maxPathSum(root *TreeNode) int {
	res := -1001
	var recursion func(n *TreeNode) int
	recursion = func(n *TreeNode) int {
		if n == nil {
			return 0
		}
		l := max(recursion(n.Left), 0)
		r := max(recursion(n.Right), 0)
		//左子树右子树都要
		LandR := n.Val + l + r
		//左子树右子树二选一
		LorR := n.Val + max(l, r)
		res = max(res, max(LandR, LorR))
		return max(LorR, 0)
	}
	recursion(root)
	return res
}

/*
125. 验证回文串
97-122 a-z 65-90 A-Z 48-57 0-9
执行用时：4 ms, 在所有 Go 提交中击败了52.19%的用户
内存消耗：2.5 MB, 在所有 Go 提交中击败了100.00%的用户
通过测试用例：480 / 480
*/
func isPalindrome(s string) bool {
	i, j := 0, len(s)-1
	for i < j {
		if s[i] > 122 || s[i] < 48 || (s[i] > 57 && s[i] < 65) || (s[i] > 90 && s[i] < 97) {
			i++
			continue
		}
		if s[j] > 122 || s[j] < 48 || (s[j] > 57 && s[j] < 65) || (s[j] > 90 && s[j] < 97) {
			j--
			continue
		}
		if s[i] >= 48 && s[i] <= 57 {
			if s[j] != s[i] {
				return false
			}
			i++
			j--
			continue
		}
		if s[i] >= 97 && s[i] <= 122 {
			if s[i] != s[j] && s[i]-32 != s[j] {
				return false
			}
			i++
			j--
			continue
		}
		if s[i] >= 65 && s[i] <= 90 {
			if s[i] != s[j] && s[i]+32 != s[j] {
				return false
			}
			i++
			j--
			continue
		}
	}
	return true
}

/*
126. 单词接龙 II todo
*/
/*func findLadders(beginWord string, endWord string, wordList []string) [][]string {

}*/

/*
127. 单词接龙 todo
*/
/*func ladderLength(beginWord string, endWord string, wordList []string) int {

}*/

/*
128. 最长连续序列
执行用时：56 ms, 在所有 Go 提交中击败了69.30%的用户
内存消耗：8 MB, 在所有 Go 提交中击败了90.62%的用户
通过测试用例：72 / 72
*/
func longestConsecutive(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	res := 1
	sort.Ints(nums)
	tmp := 1
	for i := 0; i < len(nums)-1; i++ {
		if nums[i+1]-nums[i] == 1 {
			tmp++
			if tmp > res {
				res = tmp
			}
		} else if nums[i+1] == nums[i] {
			continue
		} else {
			tmp = 1
		}
	}
	return res
}

/*
129. 求根节点到叶节点数字之和
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了68.60%的用户
通过测试用例：108 / 108
*/
func SumNumbers(root *TreeNode) int {
	var dfs func(node *TreeNode, num int)
	res := 0
	dfs = func(n *TreeNode, num int) {
		if n == nil {
			return
		}
		num = num*10 + n.Val
		if n.Left == nil && n.Right == nil {
			res += num
			return
		}
		dfs(n.Left, num)
		dfs(n.Right, num)
	}
	dfs(root, 0)
	return res
}

/*
130. 被围绕的区域
执行用时：16 ms, 在所有 Go 提交中击败了88.30%的用户
内存消耗：6.2 MB, 在所有 Go 提交中击败了54.69%的用户
通过测试用例：58 / 58
*/
func solve(board [][]byte) {
	m, n := len(board), len(board[0])
	//O与边界O相连，将符合的O全改
	var dfs func(board [][]byte, i, j int)
	dfs = func(board [][]byte, i, j int) {
		if i < 0 || j < 0 || i >= m || j >= n || board[i][j] != 'O' {
			return
		}
		board[i][j] = 'A'
		dfs(board, i+1, j)
		dfs(board, i-1, j)
		dfs(board, i, j+1)
		dfs(board, i, j-1)
	}
	for i := 0; i < m; i++ {
		dfs(board, i, 0)
		dfs(board, i, n-1)
	}
	for j := 0; j < n; j++ {
		dfs(board, 0, j)
		dfs(board, m-1, j)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if board[i][j] == 'A' {
				board[i][j] = 'O'
			} else if board[i][j] == 'O' {
				board[i][j] = 'X'
			}
		}
	}
	return
}

/*
131. 分割回文串
执行用时：264 ms, 在所有 Go 提交中击败了19.97%的用户
内存消耗：27.1 MB, 在所有 Go 提交中击败了14.17%的用户
通过测试用例：32 / 32
*/
func Partition131(s string) [][]string {
	//f[i][j]表示s[i:j+1]是否为回文
	l := len(s)
	f := make([][]bool, l)
	for i := 0; i < l; i++ {
		f[i] = make([]bool, l)
		for j := 0; j < l; j++ {
			f[i][j] = true
		}
	}

	for i := l - 1; i >= 0; i-- {
		for j := i + 1; j < l; j++ {
			f[i][j] = s[i] == s[j] && f[i+1][j-1]
		}
	}

	var res [][]string
	var tmp []string
	var dfs func(int)
	dfs = func(i int) {
		if i == l {
			res = append(res, append([]string{}, tmp...))
		}
		for j := i; j < l; j++ {
			if f[i][j] {
				tmp = append(tmp, s[i:j+1])
				dfs(j + 1)
				tmp = tmp[:len(tmp)-1]
			}
		}
	}
	dfs(0)
	return res
}

/*
132. 分割回文串 II
执行用时：24 ms, 在所有 Go 提交中击败了22.97%的用户
内存消耗：7.3 MB, 在所有 Go 提交中击败了54.50%的用户
通过测试用例：36 / 36
*/
func minCut(s string) int {
	//f[i][j]表示s[i:j+1]是否为回文
	l := len(s)
	f := make([][]bool, l)
	for i := 0; i < l; i++ {
		f[i] = make([]bool, l)
		for j := 0; j < l; j++ {
			f[i][j] = true
		}
	}

	for i := l - 1; i >= 0; i-- {
		for j := i + 1; j < l; j++ {
			f[i][j] = s[i] == s[j] && f[i+1][j-1]
		}
	}

	//r[i]为s[:i]的最小分割次数
	//r[i] = min(r[j])+1,0<=j<i
	r := make([]int, l)
	for i := range r {
		if f[0][i] {
			continue
		}
		r[i] = math.MaxInt64
		for j := 0; j < i; j++ {
			if f[j+1][i] && r[j]+1 < r[i] {
				r[i] = r[j] + 1
			}
		}
	}
	return r[l-1]
}

//type Node struct {
//	Val       int
//	Neighbors []*Node
//}

/*
133. 克隆图 todo
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.7 MB, 在所有 Go 提交中击败了81.04%的用户
通过测试用例：22 / 22
*/
/*func cloneGraph(node *Node) *Node {
	visited := map[*Node]*Node{}
	var cg func(node *Node) *Node
	cg = func(node *Node) *Node {
		if node == nil {
			return node
		}

		// 如果该节点已经被访问过了，则直接从哈希表中取出对应的克隆节点返回
		if _, ok := visited[node]; ok {
			return visited[node]
		}

		// 克隆节点，注意到为了深拷贝我们不会克隆它的邻居的列表
		cloneNode := &Node{node.Val, []*Node{}}
		// 哈希表存储
		visited[node] = cloneNode

		// 遍历该节点的邻居并更新克隆节点的邻居列表
		for _, n := range node.Neighbors {
			cloneNode.Neighbors = append(cloneNode.Neighbors, cg(n))
		}
		return cloneNode
	}
	return cg(node)
}*/

/*
134. 加油站
执行用时：64 ms, 在所有 Go 提交中击败了83.96%的用户
内存消耗：8.9 MB, 在所有 Go 提交中击败了29.80%的用户
通过测试用例：37 / 37
*/
func canCompleteCircuit(gas []int, cost []int) int {
	l := len(gas)
	res := -1
	for i := 0; i < l; {
		if gas[i] < cost[i] {
			i++
			continue
		}
		gtotal := 0
		j := 0
		for ; j < l; j++ {
			gtotal = gtotal + gas[(j+i)%l] - cost[(j+i)%l]
			if gtotal < 0 {
				break
			}
		}
		if gtotal < 0 {
			//跳转到第一个不可达的
			i += j
			continue
		}
		res = i
		break
	}
	return res
}

/*
135. 分发糖果
记前一个同学分得的糖果数量为pre
如果当前同学比上一个同学评分高，说明我们就在最近的递增序列中，直接分配给该同学pre+1个糖果即可。
否则我们就在一个递减序列中，我们直接分配给当前同学一个糖果，并把该同学所在的递减序列中所有的同学都再多分配一个糖果，以保证糖果数量还是满足条件。
当前的递减序列长度和上一个递增序列等长时，需要把最近的递增序列的最后一个同学也并进递减序列中
升序序列的结尾值为升序序列长度
执行用时：12 ms, 在所有 Go 提交中击败了96.31%的用户
内存消耗：6 MB, 在所有 Go 提交中击败了96.79%的用户
通过测试用例：48 / 48
*/
func Candy(ratings []int) int {
	pre := 1
	desc := 0
	total := 0
	asc := 1
	for i := 1; i < len(ratings); i++ {
		fmt.Println("i:", i)
		fmt.Println("total:", total)
		fmt.Println("pre:", pre)
		fmt.Println("desc:", desc)
		if ratings[i-1] < ratings[i] {
			desc = 0
			pre++
			total += pre
			asc = pre
		} else if ratings[i-1] == ratings[i] {
			pre = 1
			desc = 0
			total += pre
			asc = pre
		} else {
			desc++
			if asc == desc {
				desc++
			}
			pre = 1
			total += desc
		}
		fmt.Println("total:", total)
		fmt.Println("pre:", pre)
		fmt.Println("desc:", desc)
		fmt.Println()
	}
	return total + 1
}

/*
136. 只出现一次的数字
异或运算⊕,a⊕0=a,a⊕a=0,满足交换律和结合律
执行用时：16 ms, 在所有 Go 提交中击败了40.61%的用户
内存消耗：6 MB, 在所有 Go 提交中击败了67.92%的用户
通过测试用例：61 / 61
*/
func singleNumber(nums []int) int {
	res := 0
	l := len(nums)
	for i := 0; i < l; i++ {
		res ^= nums[i]
	}
	return res
}

/*
137. 只出现一次的数字 II
执行用时：12 ms, 在所有 Go 提交中击败了7.19%的用户
内存消耗：3.2 MB, 在所有 Go 提交中击败了100.00%的用户
通过测试用例：14 / 14
*/
func singleNumber137(nums []int) int {
	sort.Ints(nums)
	l := len(nums)
	if l == 1 {
		return nums[0]
	}
	for i := 1; i < l-1; i++ {
		if nums[i] != nums[i-1] && nums[i] != nums[i+1] {
			return nums[i]
		}
	}
	if nums[0] != nums[1] {
		return nums[0]
	}
	return nums[l-1]
}

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

/*
138. 复制带随机指针的链表
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：3.4 MB, 在所有 Go 提交中击败了47.17%的用户
通过测试用例：19 / 19
*/
func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	tmpHead := head
	//map存新旧对应关系
	m := make(map[*Node]*Node)
	r := &Node{}
	res := r
	r.Val = head.Val
	m[head] = r
	for head.Next != nil {
		tmp := &Node{
			Val:    head.Next.Val,
			Next:   nil,
			Random: nil,
		}
		r.Next = tmp
		m[head.Next] = r.Next
		r = r.Next
		head = head.Next
	}
	tmp := res
	for tmpHead != nil {
		tmp.Random = m[tmpHead.Random]
		tmp = tmp.Next
		tmpHead = tmpHead.Next
	}
	return res
}

/*
139. 单词拆分
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了57.34%的用户
通过测试用例：45 / 45
*/
func WordBreak(s string, wordDict []string) bool {
	m := make(map[string]bool)
	for i := 0; i < len(wordDict); i++ {
		m[wordDict[i]] = true
	}
	f := make([]bool, len(s)+1)
	f[0] = true
	for i := 0; i <= len(s); i++ {
		for j := 0; j < i; j++ {
			if f[j] && m[s[j:i]] {
				f[i] = true
				break
			}
		}
	}
	return f[len(s)]
}

/*
140. 单词拆分 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了76.00%的用户
通过测试用例：26 / 26
*/
func wordBreak140(s string, wordDict []string) []string {
	var res []string
	var dfs func(string, string, []string)
	dfs = func(str, tmp string, arr []string) {
		if len(str) == 0 {
			res = append(res, tmp[:len(tmp)-1])
			return
		}
		l := len(arr)
		for i := 0; i < l; i++ {
			if len(arr[i]) > len(str) {
				continue
			}
			if str[:len(arr[i])] == arr[i] {
				tmp = tmp + arr[i] + " "
				dfs(str[len(arr[i]):], tmp, arr)
				tmp = tmp[:len(tmp)-1-len(arr[i])]
			}
		}
		return
	}
	dfs(s, "", wordDict)
	return res
}

/*
141. 环形链表
执行用时：4 ms, 在所有 Go 提交中击败了98.85%的用户
内存消耗：4.2 MB, 在所有 Go 提交中击败了65.85%的用户
通过测试用例：21 / 21
*/
func hasCycle(head *ListNode) bool {
	//典中典快慢指针
	if head == nil {
		return false
	}
	slow, fast := head, head
	for fast != nil {
		slow = slow.Next
		fast = fast.Next
		if fast == nil {
			return false
		}
		fast = fast.Next
		if slow == fast {
			return true
		}
	}
	return false
}

/*
142. 环形链表 II
哈希表
执行用时：12 ms, 在所有 Go 提交中击败了25.98%的用户
内存消耗：5.8 MB, 在所有 Go 提交中击败了5.52%的用户
通过测试用例：16 / 16
快慢指针
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：3.5 MB, 在所有 Go 提交中击败了66.20%的用户
通过测试用例：16 / 16
*/
func detectCycle(head *ListNode) *ListNode {
	//哈希表
	/*if head == nil {
		return nil
	}
	m := make(map[*ListNode]int)
	for head != nil {
		m[head]++
		if m[head] == 2 {
			return head
		}
		head = head.Next
	}
	return nil*/
	//快慢指针,快 = 2*慢,假设环外长度a,慢指针在环内经过b,环长为b+c
	//2(a+b) = a+(n+1)b+nc => a = c+(n-1)(b+c)
	if head == nil {
		return nil
	}
	slow, fast, tmp := head, head, head
	for fast != nil {
		slow = slow.Next
		fast = fast.Next
		if fast == nil {
			return nil
		}
		fast = fast.Next
		if slow == fast {
			for {
				if tmp == slow {
					return tmp
				}
				tmp = tmp.Next
				slow = slow.Next
			}
		}
	}
	return nil
}

/*
143. 重排链表
寻找链表中点,右链表逆序,合并链表
123456 -> 123,654 -> 162534;1234567 -> 1234,765 -> 1726354
执行用时：8 ms, 在所有 Go 提交中击败了83.86%的用户
内存消耗：5.1 MB, 在所有 Go 提交中击败了73.43%的用户
通过测试用例：12 / 12
*/
func reorderList(head *ListNode) {
	if head == nil {
		return
	}

	//快慢指针找中点
	mid, fast := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		mid = mid.Next
		fast = fast.Next.Next
	}

	//右链表逆序
	reverse := func(n *ListNode) *ListNode {
		var pre, cur *ListNode = nil, n
		for cur != nil {
			nextTmp := cur.Next
			cur.Next = pre
			pre = cur
			cur = nextTmp
		}
		return pre
	}

	//合并
	l1 := head
	l2 := reverse(mid.Next)
	mid.Next = nil
	var tmp1, tmp2 *ListNode
	for l1 != nil && l2 != nil {
		tmp1 = l1.Next
		tmp2 = l2.Next
		l1.Next = l2
		l2.Next = tmp1
		l1 = tmp1
		l2 = tmp2
	}
	return
}

/*
144. 二叉树的前序遍历
深度优先
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了71.05%的用户
通过测试用例：69 / 69
迭代
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了99.85%的用户
通过测试用例：69 / 69
morris
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了71.05%的用户
通过测试用例：69 / 69
*/
func preorderTraversal(root *TreeNode) []int {
	var res []int
	/*var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		res = append(res, node.Val)
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)*/

	/*var iteration func(node *TreeNode)
	iteration = func(node *TreeNode) {
		if node == nil {
			return
		}
		var stack []*TreeNode
		for node != nil || len(stack) > 0 {
			for node != nil {
				res = append(res, node.Val)
				stack = append(stack, node)
				node = node.Left
			}
			node = stack[len(stack)-1].Right
			stack = stack[:len(stack)-1]
		}
	}
	iteration(root)*/

	//morris
	for root != nil {
		if root.Left != nil {
			tmp := root.Left
			for tmp.Right != nil && tmp.Right != root {
				tmp = tmp.Right
			}
			if tmp.Right == nil {
				tmp.Right = root
				res = append(res, root.Val)
				root = root.Left
			} else {
				tmp.Right = nil
				root = root.Right
			}
		} else {
			res = append(res, root.Val)
			root = root.Right
		}
	}
	return res
}

/*
145. 二叉树的后序遍历
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了80.70%的用户
通过测试用例：68 / 68
*/
func postorderTraversal(root *TreeNode) []int {
	var res []int
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		dfs(node.Right)
		res = append(res, node.Val)
	}
	dfs(root)
	return res
}

/*
146. LRU 缓存 todo
*/
/*
type LRUCache struct {

}


func Constructor(capacity int) LRUCache {

}


func (this *LRUCache) Get(key int) int {

}


func (this *LRUCache) Put(key int, value int)  {

}
*/

/*
147. 对链表进行插入排序
执行用时：4 ms, 在所有 Go 提交中击败了98.73%的用户
内存消耗：3.1 MB, 在所有 Go 提交中击败了74.73%的用户
通过测试用例：19 / 19
*/
func InsertionSortList(head *ListNode) *ListNode {
	res := head
	if head == nil || head.Next == nil {
		return res
	}
	for head.Next != nil {
		fmt.Println("head.Val:", head.Val)
		next := head.Next
		tmp := head
		if tmp.Val > next.Val {
			tmp.Next = tmp.Next.Next
			tmpHead := res
			if res.Val > next.Val {
				next.Next = res
				res = next
			} else {
				for tmpHead.Next != nil {
					if tmpHead.Next.Val > next.Val {
						break
					}
					tmpHead = tmpHead.Next
				}
				next.Next = tmpHead.Next
				tmpHead.Next = next
			}
		} else {
			head = head.Next
		}
	}
	return res
}

/*
148. 排序链表
自底而上，递归，找中点，合并
执行用时：52 ms, 在所有 Go 提交中击败了36.78%的用户
内存消耗：7.3 MB, 在所有 Go 提交中击败了61.36%的用户
通过测试用例：30 / 30
*/
func sortList(head *ListNode) *ListNode {
	var getMid func(n *ListNode) *ListNode
	getMid = func(n *ListNode) *ListNode {
		//快慢指针找中点
		mid, fast := n, n
		for fast.Next != nil && fast.Next.Next != nil {
			mid = mid.Next
			fast = fast.Next.Next
		}
		return mid
	}

	var merge func(list1 *ListNode, list2 *ListNode) *ListNode
	merge = func(list1 *ListNode, list2 *ListNode) *ListNode {
		if list1 == nil {
			return list2
		}
		if list2 == nil {
			return list1
		}

		//认为1首位更小
		if list1.Val > list2.Val {
			return merge(list2, list1)
		}

		head := &ListNode{
			Val:  0,
			Next: list1,
		}

		for list1.Next != nil && list2 != nil {
			//fmt.Println("list1.Val:", list1.Val)
			//fmt.Println("list2.Val:", list2.Val)
			if list1.Next.Val > list2.Val {
				tmp1 := list1.Next
				tmp2 := list2.Next
				list1.Next = list2
				list1.Next.Next = tmp1
				list1 = list1.Next
				list2 = tmp2
			} else {
				list1 = list1.Next
			}
		}
		if list1.Next == nil {
			list1.Next = list2
		}
		return head.Next
	}

	var recursion func(n *ListNode) *ListNode
	recursion = func(n *ListNode) *ListNode {
		if n == nil || n.Next == nil {
			return n
		}

		mid := getMid(n)
		right := mid.Next
		mid.Next = nil
		left := n

		l := recursion(left)
		r := recursion(right)

		return merge(l, r)
	}
	return recursion(head)
}

/*
149. 直线上最多的点数
先枚举所有可能出现的直线斜率，使用哈希表统计所有斜率对应的点的数量，在所有值中取个max
执行用时：8 ms, 在所有 Go 提交中击败了89.89%的用户
内存消耗：6 MB, 在所有 Go 提交中击败了74.73%的用户
通过测试用例：35 / 35
*/
func MaxPoints(points [][]int) int {
	l := len(points)
	if l <= 2 {
		return l
	}
	res := 0
	for i := 0; i < l; i++ {
		if res > l/2 || res >= l-i {
			break
		}
		m := make(map[int]int)
		for j := i + 1; j < l; j++ {
			x := points[i][0] - points[j][0]
			y := points[i][1] - points[j][1]
			if x == 0 {
				y = 1
			} else if y == 0 {
				x = 1
			} else {
				if y < 0 {
					x, y = -x, -y
				}
				g := getMaxCommonDivisor(abs(x), abs(y))
				x /= g
				y /= g
			}
			//-10000 <= xi, yi <= 10000,斜率[-20000,20000]
			m[y+x*20001]++
		}
		for _, v := range m {
			res = max(res, v+1)
		}
	}
	return res
}
func getMaxCommonDivisor(i, j int) int {
	for i != 0 {
		i, j = j%i, i
	}
	return j
}
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

/*
150. 逆波兰表达式求值
执行用时：4 ms, 在所有 Go 提交中击败了81.05%的用户
内存消耗：4 MB, 在所有 Go 提交中击败了78.01%的用户
通过测试用例：20 / 20
*/
func EvalRPN(tokens []string) int {
	var stack []int
	for i := 0; i < len(tokens); i++ {
		switch tokens[i] {
		case "+":
			stack[len(stack)-2] += stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		case "-":
			stack[len(stack)-2] -= stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		case "*":
			stack[len(stack)-2] *= stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		case "/":
			stack[len(stack)-2] /= stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		default:
			num, _ := strconv.Atoi(tokens[i])
			stack = append(stack, num)
		}
	}
	return stack[0]
}
