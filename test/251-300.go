package test

import (
	"math"
	"sort"
	"strconv"
	"strings"
)

/*
251-256,259,261,265-267,269-272.276-277,280-281。285-286,288,291,293,294,296,298 todo vip
*/

/*
257. 二叉树的所有路径
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.2 MB, 在所有 Go 提交中击败了81.49%的用户
通过测试用例：208 / 208
*/
func binaryTreePaths(root *TreeNode) []string {
	var res []string
	var dfs func(node *TreeNode, str string)
	dfs = func(node *TreeNode, str string) {
		if node == nil {
			return
		}
		if node.Left == nil && node.Right == nil {
			str += strconv.Itoa(node.Val)
			res = append(res, str)
			return
		}
		str += strconv.Itoa(node.Val) + "->"
		dfs(node.Left, str)
		dfs(node.Right, str)
	}
	dfs(root, "")
	return res
}

/*
258. 各位相加
普通方法
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了100.00%的用户
通过测试用例：1101 / 1101
数学方法
执行用时：4 ms, 在所有 Go 提交中击败了36.53%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了55.25%的用户
通过测试用例：1101 / 1101
*/
func AddDigits(num int) int {
	//for num > 9 {
	//	tmp := num
	//	r := 0
	//	for tmp > 0 {
	//		r += tmp % 10
	//		tmp /= 10
	//	}
	//	num = r
	//}
	//return num

	//数学方法
	return (num-1)%9 + 1
}

/*
260. 只出现一次的数字 III
哈希表
执行用时：4 ms, 在所有 Go 提交中击败了95.25%的用户
内存消耗：4.6 MB, 在所有 Go 提交中击败了16.27%的用户
通过测试用例：32 / 32
位运算，todo 抄的
执行用时：4 ms, 在所有 Go 提交中击败了95.25%的用户
内存消耗：3.8 MB, 在所有 Go 提交中击败了67.12%的用户
通过测试用例：32 / 32
*/
func singleNumber260(nums []int) []int {
	/*m := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		m[nums[i]]++
	}
	var res []int
	for k, v := range m {
		if v == 1 {
			res = append(res, k)
			if len(res) == 2 {
				return res
			}
		}
	}
	return res*/
	xorSum := 0
	for _, num := range nums {
		xorSum ^= num
	}
	lsb := xorSum & -xorSum
	type1, type2 := 0, 0
	for _, num := range nums {
		if num&lsb > 0 {
			type1 ^= num
		} else {
			type2 ^= num
		}
	}
	return []int{type1, type2}
}

/*
262. 行程和用户 todo sql
*/

/*
263. 丑数,只包含质因数 2、3 和 5 的正整数
执行用时：4 ms, 在所有 Go 提交中击败了18.46%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了56.38%的用户
通过测试用例：1013 / 1013
*/
func isUgly(n int) bool {
	//对n反复除以 2,3,5
	if n <= 0 {
		return false
	}
	arr := []int{2, 3, 5}
	for i := 0; i < 3; i++ {
		for n%arr[i] == 0 {
			n /= arr[i]
		}
	}
	return n == 1
}

/*
264. 丑数 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：4 MB, 在所有 Go 提交中击败了97.08%的用户
通过测试用例：596 / 596
*/
func NthUglyNumber(n int) int {
	dp := make([]int, n)
	dp[0] = 1
	p2, p3, p5 := 0, 0, 0
	for i := 1; i < n; i++ {
		t2, t3, t5 := dp[p2]*2, dp[p3]*3, dp[p5]*5
		dp[i] = min(t2, min(t3, t5))
		if dp[i] == t2 {
			p2++
		}
		if dp[i] == t3 {
			p3++
		}
		if dp[i] == t5 {
			p5++
		}
	}
	return dp[n-1]
}

/*
268. 丢失的数字
执行用时：20 ms, 在所有 Go 提交中击败了29.28%的用户
内存消耗：6 MB, 在所有 Go 提交中击败了99.77%的用户
通过测试用例：122 / 122
*/
func missingNumber(nums []int) int {
	//数学方法
	total1 := len(nums) * (len(nums) + 1) / 2
	total2 := 0
	for i := 0; i < len(nums); i++ {
		total2 += nums[i]
	}
	return total1 - total2
}

/*
273. 整数转换英文表示
最高10位，每一组最多有 3 位数，因此依次得到百位、十位、个位上的数字，生成该组的英文表示
执行用时：4 ms, 在所有 Go 提交中击败了15.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了50.00%的用户
通过测试用例：601 / 601
*/
func numberToWords(num int) string {
	singles := []string{"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"}
	teens := []string{"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"}
	tens := []string{"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"}
	thousands := []string{"", "Thousand", "Million", "Billion"}

	if num == 0 {
		return "Zero"
	}
	sb := &strings.Builder{}
	var recursion func(int)
	recursion = func(num int) {
		switch {
		case num == 0:
		case num < 10:
			sb.WriteString(singles[num])
			sb.WriteByte(' ')
		case num < 20:
			sb.WriteString(teens[num-10])
			sb.WriteByte(' ')
		case num < 100:
			sb.WriteString(tens[num/10])
			sb.WriteByte(' ')
			recursion(num % 10)
		default:
			sb.WriteString(singles[num/100])
			sb.WriteString(" Hundred ")
			recursion(num % 100)
		}
	}
	for i, unit := 3, int(1e9); i >= 0; i-- {
		if curNum := num / unit; curNum > 0 {
			num -= curNum * unit
			recursion(curNum)
			sb.WriteString(thousands[i])
			sb.WriteByte(' ')
		}
		unit /= 1000
	}
	return strings.TrimSpace(sb.String())
}

/*
274. H 指数
首先从大到小排序，从头遍历，找到第一个引用量比标号少的（注意从1开始数文章数），这个标号就是h指数；
有一种要单独判断，如果最小的引用数都比文章数大，那这人是大牛，h指数就是发的文章数
执行用时：4 ms, 在所有 Go 提交中击败了15.13%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了57.89%的用户
通过测试用例：81 / 81
*/
func HIndex(citations []int) int {
	sort.Ints(citations)
	l := len(citations)
	for i := 0; i < l; i++ {
		if citations[l-i-1] < i+1 {
			return i
		}
	}
	return len(citations)
}

/*
275. H 指数 II
二分查找
执行用时：16 ms, 在所有 Go 提交中击败了32.59%的用户
内存消耗：6.5 MB, 在所有 Go 提交中击败了63.70%的用户
通过测试用例：83 / 83
*/
func HIndex275(citations []int) int {
	n := len(citations)
	l, r := 0, len(citations)-1
	for l <= r {
		mid := (l + r) / 2
		if n-mid > citations[mid] {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return n - l
}

/*
278. 第一个错误的版本
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了9.15%的用户
通过测试用例：24 / 24
*/
func FirstBadVersion(n int) int {
	i, j := 1, n
	for i < j {
		mid := (i + j) / 2
		if isBadVersion(mid) {
			j = mid
		} else {
			i = mid + 1
		}
	}
	return i
}
func isBadVersion(version int) bool {
	if version < 4 {
		return false
	}
	return true
}

/*
279. 完全平方数
执行用时：20 ms, 在所有 Go 提交中击败了83.12%的用户
内存消耗：6.1 MB, 在所有 Go 提交中击败了83.80%的用户
通过测试用例：588 / 588
*/
func numSquares(n int) int {
	//四平方和定理
	/*if isPerfectSquare(n) {
		return 1
	}
	if checkAnswer4(n) {
		return 4
	}
	for i := 1; i*i <= n; i++ {
		j := n - i*i
		if isPerfectSquare(j) {
			return 2
		}
	}
	return 3*/

	//动态规划
	m := make([]int, n+1)
	for i := 1; i <= n; i++ {
		tmp := math.MaxInt32
		for j := 1; j*j <= i; j++ {
			tmp = min(tmp, m[i-j*j])
		}
		m[i] = tmp + 1
	}
	return m[n]
}

// 判断是否为完全平方数
func isPerfectSquare(x int) bool {
	y := int(math.Sqrt(float64(x)))
	return y*y == x
}

// 判断是否能表示为 4^k*(8m+7)
func checkAnswer4(x int) bool {
	for x%4 == 0 {
		x /= 4
	}
	return x%8 == 7
}

/*
282. 给表达式添加运算符
执行用时：96 ms, 在所有 Go 提交中击败了7.32%的用户
内存消耗：6.6 MB, 在所有 Go 提交中击败了17.07%的用户
通过测试用例：23 / 23
*/
func AddOperators(num string, target int) []string {
	l := len(num)
	var res []string
	//mul 为表达式最后一个连乘串的计算结果
	var dfs func(str string, sum, index, mul int)
	dfs = func(str string, sum, index, mul int) {
		if index == l {
			if sum == target {
				res = append(res, str)
			}
			return
		}
		for i, tmp := index, 0; i < l && (i == index || num[index] != '0'); i++ {
			tmp = tmp*10 + int(num[i]-'0')
			if index == 0 {
				dfs(str+num[index:i+1], tmp, i+1, tmp)
			} else {
				dfs(str+"+"+num[index:i+1], sum+tmp, i+1, tmp)
				dfs(str+"-"+num[index:i+1], sum-tmp, i+1, 0-tmp)
				dfs(str+"*"+num[index:i+1], sum-mul+mul*tmp, i+1, tmp*mul)
			}
		}
	}
	dfs("", 0, 0, 0)
	return res
}

/*
283. 移动零
执行用时：20 ms, 在所有 Go 提交中击败了74.45%的用户
内存消耗：6.5 MB, 在所有 Go 提交中击败了58.39%的用户
通过测试用例：74 / 74
*/
func MoveZeroes(nums []int) {
	i, j, l := 0, 0, len(nums)
	for j < l {
		if nums[j] != 0 {
			nums[i], nums[j] = nums[j], nums[i]
			i++
		}
		j++
	}
}

/*
284. 顶端迭代器
执行用时：4 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了64.71%的用户
通过测试用例：14 / 14
*/
/*type PeekingIterator struct {
	iter     *Iterator
	_hasNext bool
	_next    int
}

func Constructor284(iter *Iterator) *PeekingIterator {
	return &PeekingIterator{iter, iter.hasNext(), iter.next()}
}

func (it *PeekingIterator) hasNext() bool {
	return it._hasNext
}

func (it *PeekingIterator) next() int {
	ret := it._next
	it._hasNext = it.iter.hasNext()
	if it._hasNext {
		it._next = it.iter.next()
	}
	return ret
}

func (it *PeekingIterator) peek() int {
	return it._next
}*/

/*
287. 寻找重复数
对 nums 数组建图，有环
执行用时：104 ms, 在所有 Go 提交中击败了12.98%的用户
内存消耗：8.2 MB, 在所有 Go 提交中击败了39.81%的用户
通过测试用例：58 / 58
*/
func FindDuplicate(nums []int) int {
	i, j := 0, 0
	for i, j = nums[i], nums[nums[j]]; i != j; i, j = nums[i], nums[nums[j]] {

	}
	i = 0
	for i != j {
		i, j = nums[i], nums[j]
	}
	return i
}

/*
289. 生命游戏
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了88.89%的用户
通过测试用例：22 / 22
*/
func GameOfLife(board [][]int) {
	var getValue func(i, j int) int
	getValue = func(i, j int) int {
		if i < 0 || j < 0 || i >= len(board) || j >= len(board[0]) {
			return 0
		}
		if board[i][j] >= 1 {
			return 1
		}
		return 0
	}
	var getAliveNum func(i, j int) int
	getAliveNum = func(i, j int) int {
		return getValue(i-1, j-1) + getValue(i, j-1) + getValue(i+1, j-1) + getValue(i-1, j) +
			getValue(i+1, j) + getValue(i-1, j+1) + getValue(i, j+1) + getValue(i+1, j+1)
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if board[i][j] >= 1 {
				//周围2，3活，其他死;死临时变为2
				tmp := getAliveNum(i, j)
				if tmp != 2 && tmp != 3 {
					board[i][j] = 2
				}
			} else {
				//周围3活，其他死;活临时变为-1
				tmp := getAliveNum(i, j)
				if tmp == 3 {
					board[i][j] = -1
				}
			}
		}
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if board[i][j] > 1 {
				board[i][j] = 0
			} else if board[i][j] < 0 {
				board[i][j] = 1
			}
		}
	}
}

/*
290. 单词规律
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了71.19%的用户
通过测试用例：36 / 36
*/
func wordPattern(pattern string, s string) bool {
	strArr := strings.Fields(s)
	if len(strArr) != len(pattern) {
		return false
	}
	m1 := make(map[string]string)
	m2 := make(map[string]string)
	for i := 0; i < len(pattern); i++ {
		_, ok1 := m1[pattern[i:i+1]]
		_, ok2 := m2[strArr[i]]
		if ok1 && ok2 {
			if strArr[i] != m1[pattern[i:i+1]] || pattern[i:i+1] != m2[strArr[i]] {
				return false
			}
		} else if !ok1 && !ok2 {
			m1[pattern[i:i+1]] = strArr[i]
			m2[strArr[i]] = pattern[i : i+1]
		} else {
			return false
		}
	}
	return true
}

/*
292. Nim 游戏
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了82.14%的用户
通过测试用例：60 / 60
*/
func canWinNim(n int) bool {
	return n%4 != 0
}

/*
295. 数据流的中位数 todo
*/

/*
297. 二叉树的序列化与反序列化
执行用时：8 ms, 在所有 Go 提交中击败了87.13%的用户
内存消耗：7.1 MB, 在所有 Go 提交中击败了40.63%的用户
通过测试用例：52 / 52
*/
type Codec struct {
}

func Constructor297() Codec {
	return Codec{}
}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	if root == nil {
		return "x"
	}
	return strconv.Itoa(root.Val) + "," + this.serialize(root.Left) + "," + this.serialize(root.Right)
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
	arr := strings.Split(data, ",")
	var build func(s *[]string) *TreeNode
	build = func(s *[]string) *TreeNode {
		cur := (*s)[0]
		*s = (*s)[1:]
		if cur == "x" {
			return nil
		}
		i, _ := strconv.Atoi(cur)
		return &TreeNode{
			Val:   i,
			Left:  build(s),
			Right: build(s),
		}
	}
	return build(&arr)
}

/*
299. 猜数字游戏
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了13.33%的用户
通过测试用例：152 / 152
*/
func getHint(secret string, guess string) string {
	l := len(secret)
	a := 0
	m1 := make(map[int]int)
	m2 := make(map[int]int)
	for i := 0; i < l; i++ {
		if secret[i] == guess[i] {
			a++
		} else {
			m1[int(secret[i]-'0')]++
			m2[int(guess[i]-'0')]++
		}
	}
	b := 0
	for i := 0; i < 10; i++ {
		b += min(m1[i], m2[i])
	}
	return strconv.Itoa(a) + "A" + strconv.Itoa(b) + "B"
}

/*
300. 最长递增子序列
执行用时：56 ms, 在所有 Go 提交中击败了59.96%的用户
内存消耗：3.6 MB, 在所有 Go 提交中击败了15.31%的用户
通过测试用例：54 / 54
*/
func LengthOfLIS(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	dp := make([]int, len(nums))
	dp[0] = 1
	res := 1
	for i := 1; i < len(nums); i++ {
		dp[i] = 1
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
		res = max(res, dp[i])
	}
	return res
}
