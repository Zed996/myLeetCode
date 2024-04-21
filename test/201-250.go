package test

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

/*
201. 数字范围按位与
执行用时：16 ms, 在所有 Go 提交中击败了34.12%的用户
内存消耗：5 MB, 在所有 Go 提交中击败了98.82%的用户
通过测试用例：8268 / 8268
*/
func rangeBitwiseAnd(left int, right int) int {
	//Brian Kernighan 算法,用于清除二进制串中最右边的 1
	for left < right {
		right &= right - 1
	}
	return right
}

/*
202. 快乐数
数学方法：4→16→37→58→89→145→42→20→4 或 1→1，最终只有两种情况
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了61.21%的用户
通过测试用例：404 / 404
*/
func isHappy(n int) bool {
	var getNext func(i int) int
	getNext = func(i int) int {
		r := 0
		for i > 0 {
			r += (i % 10) * (i % 10)
			i /= 10
		}
		return r
	}
	//快慢指针判断环
	fast, slow := getNext(getNext(n)), getNext(n)
	for fast != 1 {
		if fast == slow {
			return false
		}
		fast = getNext(getNext(fast))
		slow = getNext(slow)
	}
	return true
}

/*
203. 移除链表元素
执行用时：4 ms, 在所有 Go 提交中击败了97.84%的用户
内存消耗：4.5 MB, 在所有 Go 提交中击败了63.05%的用户
通过测试用例：66 / 66
*/
func removeElements(head *ListNode, val int) *ListNode {
	if head == nil {
		return head
	}
	tmp := &ListNode{
		Val:  0,
		Next: head,
	}
	r := tmp
	for r != nil && r.Next != nil {
		if r.Next.Val == val {
			r.Next = r.Next.Next
		} else {
			r = r.Next
		}
	}
	return tmp.Next
}

/*
204. 计数质数
线性筛,维护质数数组,遍历标记质数集合中的数与x相乘的数
执行用时：140 ms, 在所有 Go 提交中击败了20.93%的用户
内存消耗：17.2 MB, 在所有 Go 提交中击败了21.78%的用户
通过测试用例：66 / 66
*/
func countPrimes(n int) int {
	if n < 2 {
		return 0
	}
	b := make([]bool, n)
	for i := 0; i < n; i++ {
		b[i] = true
	}
	var primes []int
	for i := 2; i < n; i++ {
		if b[i] {
			primes = append(primes, i)
		}
		for j := 0; j < len(primes); j++ {
			if i*primes[j] >= n {
				break
			}
			b[i*primes[j]] = false
			//每个合数被标记一次
			if i%primes[j] == 0 {
				break
			}
		}
	}
	return len(primes)
}

/*
205. 同构字符串
执行用时：4 ms, 在所有 Go 提交中击败了69.73%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了65.85%的用户
通过测试用例：43 / 43
*/
func IsIsomorphic(s string, t string) bool {
	m1 := make(map[byte]byte)
	m2 := make(map[byte]byte)
	l := len(s)
	for i := 0; i < l; i++ {
		_, ok := m1[s[i]]
		if !ok {
			m1[s[i]] = t[i]
		} else if m1[s[i]] != t[i] {
			return false
		}
		_, ok = m2[t[i]]
		if !ok {
			m2[t[i]] = s[i]
		} else if m2[t[i]] != s[i] {
			return false
		}
	}
	return true
}

/*
206. 反转链表
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了99.94%的用户
通过测试用例：28 / 28
*/
func ReverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	n1 := head
	n2 := head.Next
	head.Next = nil
	for n2 != nil {
		if n2.Next == nil {
			n2.Next = n1
			return n2
		} else {
			tmp := n2.Next
			n2.Next = n1
			n1 = n2
			n2 = tmp
		}
	}
	return n2
}

/*
207. 课程表
判断有向无环图；广度优先，每次删除一个入度为0的，修改相关联的点的入度；也可以深度优先
执行用时：8 ms, 在所有 Go 提交中击败了91.75%的用户
内存消耗：6.2 MB, 在所有 Go 提交中击败了39.89%的用户
通过测试用例：52 / 52
*/
func CanFinish(numCourses int, prerequisites [][]int) bool {
	in := make(map[int]int)
	m := make(map[int][]int)
	for i := 0; i < numCourses; i++ {
		in[i] = 0
	}
	for i := 0; i < len(prerequisites); i++ {
		in[prerequisites[i][0]]++
		m[prerequisites[i][1]] = append(m[prerequisites[i][1]], prerequisites[i][0])
	}
	var zeroInArr []int
	for k, v := range in {
		if v == 0 {
			zeroInArr = append(zeroInArr, k)
		}
	}
	//fmt.Println(m)

	for count := 0; count < numCourses; count++ {
		//fmt.Println(zeroInArr)
		//fmt.Println(in)
		if len(zeroInArr) == 0 {
			return false
		}
		tmp := zeroInArr[0]
		zeroInArr = zeroInArr[1:]
		for _, v := range m[tmp] {
			//fmt.Println("v:", v)
			in[v]--
			if in[v] == 0 {
				zeroInArr = append(zeroInArr, v)
			}
		}
	}
	if len(zeroInArr) > 0 {
		return false
	}
	return true
}

/*
208. 实现 Trie (前缀树)
执行用时：44 ms, 在所有 Go 提交中击败了87.18%的用户
内存消耗：18 MB, 在所有 Go 提交中击败了19.31%的用户
通过测试用例：16 / 16
*/
type Trie struct {
	children [26]*Trie
	isEnd    bool
}

func Constructor208() Trie {
	return Trie{}
}
func (this *Trie) Insert(word string) {
	node := this
	for i := 0; i < len(word); i++ {
		tmp := word[i] - 'a'
		if node.children[tmp] == nil {
			node.children[tmp] = &Trie{}
		}
		node = node.children[tmp]
	}
	node.isEnd = true
}
func (this *Trie) Search(word string) bool {
	node := this
	for i := 0; i < len(word); i++ {
		tmp := word[i] - 'a'
		if node.children[tmp] == nil {
			return false
		}
		node = node.children[tmp]
	}
	return node.isEnd
}
func (this *Trie) StartsWith(prefix string) bool {
	node := this
	for i := 0; i < len(prefix); i++ {
		tmp := prefix[i] - 'a'
		if node.children[tmp] == nil {
			return false
		}
		node = node.children[tmp]
	}
	return true
}

/*
209. 长度最小的子数组
滑动窗口
执行用时：28 ms, 在所有 Go 提交中击败了49.09%的用户
内存消耗：8.1 MB, 在所有 Go 提交中击败了45.07%的用户
通过测试用例：20 / 20
*/
func minSubArrayLen(target int, nums []int) int {
	l, r := 0, 0
	res := math.MaxInt
	count := nums[0]
	for l < len(nums) {
		if count >= target {
			res = min(r-l+1, res)
			count -= nums[l]
			l++
		} else {
			if r == len(nums)-1 {
				break
			}
			r++
			count += nums[r]
		}
	}
	if res == math.MaxInt {
		return 0
	}
	return res
}

/*
210. 课程表 II
执行用时：12 ms, 在所有 Go 提交中击败了41.24%的用户
内存消耗：6.4 MB, 在所有 Go 提交中击败了26.42%的用户
通过测试用例：45 / 45
*/
func findOrder(numCourses int, prerequisites [][]int) []int {
	in := make(map[int]int)
	m := make(map[int][]int)
	for i := 0; i < numCourses; i++ {
		in[i] = 0
	}
	for i := 0; i < len(prerequisites); i++ {
		in[prerequisites[i][0]]++
		m[prerequisites[i][1]] = append(m[prerequisites[i][1]], prerequisites[i][0])
	}
	var zeroInArr []int
	var res []int
	for k, v := range in {
		if v == 0 {
			zeroInArr = append(zeroInArr, k)
			res = append(res, k)
		}
	}

	for count := 0; count < numCourses; count++ {
		//fmt.Println(zeroInArr)
		//fmt.Println(in)
		if len(zeroInArr) == 0 {
			return []int{}
		}
		tmp := zeroInArr[0]
		zeroInArr = zeroInArr[1:]
		for _, v := range m[tmp] {
			//fmt.Println("v:", v)
			in[v]--
			if in[v] == 0 {
				zeroInArr = append(zeroInArr, v)
				res = append(res, v)
			}
		}
	}
	if len(zeroInArr) > 0 {
		return []int{}
	}
	return res
}

/*
211. 添加与搜索单词 - 数据结构设计
参考208
执行用时：504 ms, 在所有 Go 提交中击败了84.95%的用户
内存消耗：60.4 MB, 在所有 Go 提交中击败了69.36%的用户
通过测试用例：29 / 29
*/
type WordDictionary struct {
	children [26]*WordDictionary
	isEnd    bool
}

func Constructor211() WordDictionary {
	return WordDictionary{}
}
func (this *WordDictionary) AddWord(word string) {
	node := this
	for i := 0; i < len(word); i++ {
		tmp := word[i] - 'a'
		if node.children[tmp] == nil {
			node.children[tmp] = &WordDictionary{}
		}
		node = node.children[tmp]
	}
	node.isEnd = true
}
func (this *WordDictionary) Search(word string) bool {
	var dfs func(int, *WordDictionary) bool
	dfs = func(i int, dictionary *WordDictionary) bool {
		if i == len(word) {
			return dictionary.isEnd
		}
		if word[i] == '.' {
			for j := 0; j < 26; j++ {
				child := dictionary.children[j]
				if child != nil && dfs(i+1, child) {
					return true
				}
			}
		} else {
			tmp := word[i] - 'a'
			if dictionary.children[tmp] == nil {
				return false
			}
			dictionary = dictionary.children[tmp]
			return dfs(i+1, dictionary)
		}
		return false
	}
	return dfs(0, this)
}

/*
212. 单词搜索 II todo 抄的
执行用时：220 ms, 在所有 Go 提交中击败了57.74%的用户
内存消耗：4.2 MB, 在所有 Go 提交中击败了75.31%的用户
通过测试用例：63 / 63
*/
func findWords(board [][]byte, words []string) []string {
	var dirs = []struct{ x, y int }{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	t := &Trie212{}
	for _, word := range words {
		t.Insert212(word)
	}

	m, n := len(board), len(board[0])
	seen := map[string]bool{}

	var dfs func(node *Trie212, x, y int)
	dfs = func(node *Trie212, x, y int) {
		ch := board[x][y]
		node = node.children[ch-'a']
		if node == nil {
			return
		}

		if node.word != "" {
			seen[node.word] = true
		}

		board[x][y] = '#'
		for _, d := range dirs {
			nx, ny := x+d.x, y+d.y
			if 0 <= nx && nx < m && 0 <= ny && ny < n && board[nx][ny] != '#' {
				dfs(node, nx, ny)
			}
		}
		board[x][y] = ch
	}

	for i, row := range board {
		for j := range row {
			dfs(t, i, j)
		}
	}
	ans := make([]string, 0, len(seen))
	for s := range seen {
		ans = append(ans, s)
	}
	return ans
}

type Trie212 struct {
	children [26]*Trie212
	word     string
}

func (t *Trie212) Insert212(word string) {
	node := t
	for _, ch := range word {
		ch -= 'a'
		if node.children[ch] == nil {
			node.children[ch] = &Trie212{}
		}
		node = node.children[ch]
	}
	node.word = word
}

/*
213. 打家劫舍 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了87.85%的用户
通过测试用例：75 / 75
*/
func rob213(nums []int) int {
	l := len(nums)
	if l == 1 {
		return nums[0]
	}
	return max(rob(nums[1:]), rob(nums[:len(nums)-1]))
}

/*
214. 最短回文串
由于只能在s前添加，找s中以s[0]开头的最长回文串，然后倒转剩余部分拼接到s前--辣鸡方法
执行用时：140 ms, 在所有 Go 提交中击败了12.03%的用户
内存消耗：2.8 MB, 在所有 Go 提交中击败了72.93%的用户
通过测试用例：121 / 121
*/
func ShortestPalindrome(s string) string {
	l := len(s)
	if l == 0 {
		return ""
	}
	m := 1
	for i := l; i > 0; i-- {
		if isPalindrome214(s[:i]) {
			m = i
			break
		}
	}
	fmt.Println(m)
	return reverseString(s[m:]) + s
}
func isPalindrome214(s string) bool {
	if len(s) == 0 {
		return false
	}
	l, r := 0, len(s)-1
	for l < r {
		if s[l] == s[r] {
			l++
			r--
		} else {
			return false
		}
	}
	return true
}
func reverseString(str string) string {
	strArr := []rune(str)
	for i := 0; i < len(strArr)/2; i++ {
		strArr[len(strArr)-1-i], strArr[i] = strArr[i], strArr[len(strArr)-1-i]
	}
	return string(strArr)
}

/*
215. 数组中的第K个最大元素
快速选择算法
执行用时：68 ms, 在所有 Go 提交中击败了26.99%的用户
内存消耗：8.3 MB, 在所有 Go 提交中击败了27.77%的用户
通过测试用例：39 / 39
*/
func FindKthLargest(nums []int, k int) int {
	if len(nums) == 1 {
		return nums[0]
	}

	l, r, n := 0, 1, len(nums)
	rand.Seed(time.Now().UnixNano())
	tmp := rand.Intn(n - 1)
	nums[tmp], nums[n-1] = nums[n-1], nums[tmp]
	for ; r < n; r++ {
		if nums[r] <= nums[n-1] {
			if nums[l] > nums[r] {
				nums[l], nums[r] = nums[r], nums[l]
			}
		}
		if nums[l] <= nums[n-1] {
			l++
		}
	}
	nums[l], nums[r-1] = nums[r-1], nums[l]
	tmp = l - 1
	if tmp != n-k {
		if tmp > n-k {
			//左侧
			return FindKthLargest(nums[:tmp], tmp+k-n)
		} else {
			//右侧
			return FindKthLargest(nums[tmp+1:], k)
		}
	}
	return nums[tmp]
}

/*
216. 组合总和 III
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了55.14%的用户
通过测试用例：18 / 18
*/
func CombinationSum3(k int, n int) [][]int {
	var res [][]int
	var arr []int
	var dfs func(index, num, total int)
	dfs = func(index, num, total int) {
		if total == 0 && num == 0 {
			res = append(res, append([]int{}, arr...))
			return
		}
		if total < 0 || index > 9 || num < 0 {
			return
		}

		dfs(index+1, num, total)

		arr = append(arr, index)
		dfs(index+1, num-1, total-index)
		arr = arr[:len(arr)-1]

	}
	dfs(1, k, n)
	return res
}

/*
217. 存在重复元素
执行用时：64 ms, 在所有 Go 提交中击败了63.21%的用户
内存消耗：8.8 MB, 在所有 Go 提交中击败了34.81%的用户
通过测试用例：70 / 70
*/
func containsDuplicate(nums []int) bool {
	m := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		if m[nums[i]] > 0 {
			return true
		}
		m[nums[i]]++
	}
	return false
}

/*
218. 天际线问题 todo
每个高度不同的矩形左上端点组成的数组
*/
/*func getSkyline(buildings [][]int) [][]int {

}*/

/*
219. 存在重复元素 II
执行用时：120 ms, 在所有 Go 提交中击败了20.68%的用户
内存消耗：8.6 MB, 在所有 Go 提交中击败了68.59%的用户
通过测试用例：52 / 52
*/
func ContainsNearbyDuplicate(nums []int, k int) bool {
	m := make(map[int]bool)
	for i := 0; i < len(nums); i++ {
		if i > k {
			delete(m, nums[i-k-1])
		}
		if v, ok := m[nums[i]]; ok && v {
			return true
		}
		m[nums[i]] = true
	}
	return false
}

/*
220. 存在重复元素 III
类似桶排序，按照元素的大小进行分桶，维护一个滑动窗口内的元素对应的元素
整数 x 表示为 x=(t+1)×a+b(0≤b≤t) 的形式，这样 x 即归属于编号为 a 的桶
执行用时：16 ms, 在所有 Go 提交中击败了95.59%的用户
内存消耗：6.2 MB, 在所有 Go 提交中击败了40.09%的用户
通过测试用例：54 / 54
*/
func containsNearbyAlmostDuplicate(nums []int, k, t int) bool {
	mp := map[int]int{}
	for i, x := range nums {
		id := getID(x, t+1)
		if _, has := mp[id]; has {
			return true
		}
		if y, has := mp[id-1]; has && abs(x-y) <= t {
			return true
		}
		if y, has := mp[id+1]; has && abs(x-y) <= t {
			return true
		}
		mp[id] = x
		if i >= k {
			delete(mp, getID(nums[i-k], t+1))
		}
	}
	return false
}
func getID(x, w int) int {
	if x >= 0 {
		return x / w
	}
	return (x+1)/w - 1
}

/*
221. 最大正方形
动态规划
matrix[i][j] = 1->dp(i,j) = min(dp(i−1,j),dp(i−1,j−1),dp(i,j−1))+1
matrix[i][j] = 0->dp(i,j) = 0
执行用时：4 ms, 在所有 Go 提交中击败了95.63%的用户
内存消耗：6.4 MB, 在所有 Go 提交中击败了7.72%的用户
通过测试用例：77 / 77
*/
func MaximalSquare(matrix [][]byte) int {
	m, n := len(matrix), len(matrix[0])
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	dp[0][0] = 0
	dp[1][0] = 0
	dp[0][1] = 0
	max := 1
	if matrix[0][0] == '0' {
		max = 0
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if matrix[i-1][j-1] == '0' {
				dp[i][j] = 0
			} else {
				dp[i][j] = min(dp[i-1][j], min(dp[i-1][j-1], dp[i][j-1])) + 1
				if dp[i][j] > max {
					max = dp[i][j]
				}
			}
		}
	}
	return max * max
}

/*
222. 完全二叉树的节点个数
判断左右高度
执行用时：16 ms, 在所有 Go 提交中击败了35.10%的用户
内存消耗：7.1 MB, 在所有 Go 提交中击败了50.93%的用户
通过测试用例：18 / 18
*/
func CountNodes(root *TreeNode) int {
	var getHeight func(node *TreeNode) int
	getHeight = func(node *TreeNode) int {
		res := 0
		for node != nil {
			res++
			node = node.Left
		}
		return res
	}
	res := 0
	for root != nil {
		l, r := getHeight(root.Left), getHeight(root.Right)
		if l != r {
			res += int(MyPow(2, r))
			root = root.Left
		} else {
			res += int(MyPow(2, l))
			root = root.Right
		}
	}
	return res
}

/*
223. 矩形面积
执行用时：8 ms, 在所有 Go 提交中击败了81.91%的用户
内存消耗：6 MB, 在所有 Go 提交中击败了64.89%的用户
通过测试用例：3080 / 3080
*/
func computeArea(ax1 int, ay1 int, ax2 int, ay2 int, bx1 int, by1 int, bx2 int, by2 int) int {
	areaA := (ax2 - ax1) * (ay2 - ay1)
	areaB := (bx2 - bx1) * (by2 - by1)
	overlapWidth := min(ax2, bx2) - max(ax1, bx1)
	overlapHeight := min(ay2, by2) - max(ay1, by1)
	overlapArea := max(overlapWidth, 0) * max(overlapHeight, 0)
	return areaA + areaB - overlapArea
}

/*
224. 基本计算器
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：7.6 MB, 在所有 Go 提交中击败了12.33%的用户
通过测试用例：42 / 42
*/
func Calculate(s string) int {
	s = strings.Replace(s, "(-", "(0-", -1)
	s = strings.Replace(s, " ", "", -1)
	if s[0] == '-' {
		s = "0" + s
	}
	var ops []string
	var nums []int
	isNum := false

	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '+':
			ops = append(ops, "+")
			isNum = false
		case '-':
			ops = append(ops, "-")
			isNum = false
		case '(':
			ops = append(ops, "(")
			isNum = false
		case ')':
			for ops[len(ops)-1] != "(" {
				if ops[len(ops)-2] == "-" {
					nums[len(nums)-2] = 0 - nums[len(nums)-2]
					ops[len(ops)-2] = "+"
				}
				if ops[len(ops)-1] == "+" {
					nums[len(nums)-2] = nums[len(nums)-2] + nums[len(nums)-1]
					nums = nums[:len(nums)-1]
				} else {
					nums[len(nums)-2] = nums[len(nums)-2] - nums[len(nums)-1]
					nums = nums[:len(nums)-1]
				}
				ops = ops[:len(ops)-1]
			}
			ops = ops[:len(ops)-1]
			isNum = false
		default:
			if isNum {
				nums[len(nums)-1] *= 10
				nums[len(nums)-1] += int(s[i] - '0')
			} else {
				nums = append(nums, int(s[i]-'0'))
			}
			isNum = true
		}
	}
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		if ops[i-1] == "+" {
			res += nums[i]
		} else {
			res -= nums[i]
		}
	}
	return res
}

/*
225. 用队列实现栈
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了13.66%的用户
通过测试用例：17 / 17
*/
type MyStack struct {
	queue []int
}

func Constructor225() MyStack {
	return MyStack{}

}
func (this *MyStack) Push(x int) {
	this.queue = append(this.queue, x)
}
func (this *MyStack) Pop() int {
	tmp := this.queue[len(this.queue)-1]
	this.queue = this.queue[:len(this.queue)-1]
	return tmp
}
func (this *MyStack) Top() int {
	return this.queue[len(this.queue)-1]
}
func (this *MyStack) Empty() bool {
	return len(this.queue) == 0
}

/*
226. 翻转二叉树
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了99.78%的用户
通过测试用例：77 / 77
*/
func invertTree(root *TreeNode) *TreeNode {
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		node.Left, node.Right = node.Right, node.Left
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	return root
}

/*
227. 基本计算器 II
执行用时：4 ms, 在所有 Go 提交中击败了80.12%的用户
内存消耗：9.9 MB, 在所有 Go 提交中击败了9.02%的用户
通过测试用例：109 / 109
*/
func Calculate227(s string) int {
	s = strings.Replace(s, " ", "", -1)
	var ops []string
	var nums []int
	isNum := false

	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '+':
			if len(ops) > 0 && ops[len(ops)-1] == "*" {
				nums[len(nums)-2] *= nums[len(nums)-1]
				nums = nums[:len(nums)-1]
				ops = ops[:len(ops)-1]
			} else if len(ops) > 0 && ops[len(ops)-1] == "/" {
				nums[len(nums)-2] /= nums[len(nums)-1]
				nums = nums[:len(nums)-1]
				ops = ops[:len(ops)-1]
			}
			ops = append(ops, "+")
			isNum = false
		case '-':
			if len(ops) > 0 && ops[len(ops)-1] == "*" {
				nums[len(nums)-2] *= nums[len(nums)-1]
				nums = nums[:len(nums)-1]
				ops = ops[:len(ops)-1]
			} else if len(ops) > 0 && ops[len(ops)-1] == "/" {
				nums[len(nums)-2] /= nums[len(nums)-1]
				nums = nums[:len(nums)-1]
				ops = ops[:len(ops)-1]
			}
			ops = append(ops, "-")
			isNum = false
		case '*':
			if len(ops) > 0 && ops[len(ops)-1] == "*" {
				nums[len(nums)-2] *= nums[len(nums)-1]
				nums = nums[:len(nums)-1]
				ops = ops[:len(ops)-1]
			} else if len(ops) > 0 && ops[len(ops)-1] == "/" {
				nums[len(nums)-2] /= nums[len(nums)-1]
				nums = nums[:len(nums)-1]
				ops = ops[:len(ops)-1]
			}
			ops = append(ops, "*")
			isNum = false
		case '/':
			if len(ops) > 0 && ops[len(ops)-1] == "*" {
				nums[len(nums)-2] *= nums[len(nums)-1]
				nums = nums[:len(nums)-1]
				ops = ops[:len(ops)-1]
			} else if len(ops) > 0 && ops[len(ops)-1] == "/" {
				nums[len(nums)-2] /= nums[len(nums)-1]
				nums = nums[:len(nums)-1]
				ops = ops[:len(ops)-1]
			}
			ops = append(ops, "/")
			isNum = false
		default:
			if isNum {
				nums[len(nums)-1] *= 10
				nums[len(nums)-1] += int(s[i] - '0')
			} else {
				nums = append(nums, int(s[i]-'0'))
			}
			isNum = true
		}
	}
	if len(ops) > 0 && ops[len(ops)-1] == "*" {
		nums[len(nums)-2] *= nums[len(nums)-1]
		nums = nums[:len(nums)-1]
		ops = ops[:len(ops)-1]
	} else if len(ops) > 0 && ops[len(ops)-1] == "/" {
		nums[len(nums)-2] /= nums[len(nums)-1]
		nums = nums[:len(nums)-1]
		ops = ops[:len(ops)-1]
	}
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		if ops[i-1] == "+" {
			res += nums[i]
		} else {
			res -= nums[i]
		}
	}
	return res
}

/*
228. 汇总区间
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了75.24%的用户
通过测试用例：29 / 29
*/
func summaryRanges(nums []int) []string {
	l := len(nums)
	if l == 0 {
		return []string{}
	}
	nums = append(nums, math.MinInt)
	var res []string
	start := nums[0]
	for i := 1; i <= l; i++ {
		if nums[i]-nums[i-1] > 1 || nums[i]-nums[i-1] < 0 {
			if start != nums[i-1] {
				res = append(res, strconv.Itoa(start)+"->"+strconv.Itoa(nums[i-1]))
			} else {
				res = append(res, strconv.Itoa(start))
			}
			start = nums[i]
		}
	}
	return res
}

/*
229. 多数元素 II
Boyer-Moore 投票算法,参考169. 多数元素
执行用时：12 ms, 在所有 Go 提交中击败了20.92%的用户
内存消耗：4.9 MB, 在所有 Go 提交中击败了14.80%的用户
通过测试用例：83 / 83
*/
func MajorityElement229(nums []int) []int {
	count1, count2 := 0, 0
	candidate1, candidate2 := 0, 0

	for i := 0; i < len(nums); i++ {
		if count1 > 0 && nums[i] == candidate1 {
			count1++
		} else if count2 > 0 && nums[i] == candidate2 {
			count2++
		} else if count1 == 0 {
			candidate1 = nums[i]
			count1++
		} else if count2 == 0 {
			candidate2 = nums[i]
			count2++
		} else {
			count1--
			count2--
		}
	}

	n1, n2 := 0, 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == candidate1 && count1 > 0 {
			n1++
		} else if nums[i] == candidate2 && count2 > 0 {
			n2++
		}
	}

	var res []int
	if n1 > len(nums)/3 && count1 > 0 {
		res = append(res, candidate1)
	}
	if n2 > len(nums)/3 && count2 > 0 {
		res = append(res, candidate2)
	}
	return res
}

/*
230. 二叉搜索树中第K小的元素
执行用时：12 ms, 在所有 Go 提交中击败了14.31%的用户
内存消耗：6.1 MB, 在所有 Go 提交中击败了97.67%的用户
通过测试用例：93 / 93
*/
func KthSmallest(root *TreeNode, k int) int {
	var res int
	count := 0
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		count++
		if count == k {
			res = node.Val
			return
		}
		dfs(node.Right)
	}
	dfs(root)
	return res
}

/*
231. 2 的幂
位运算：n&(n-1) == 0或n&-n == n，都代表n是2的幂
n&(n-1)：移除 n 二进制表示的最低位 1
n&-n：获取 n 二进制表示的最低位的 1
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了57.83%的用户
通过测试用例：1108 / 1108
*/
func isPowerOfTwo(n int) bool {
	return n > 0 && n&(n-1) == 0 && n&-n == n
}

/*
232. 用栈实现队列
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了99.83%的用户
通过测试用例：22 / 22
*/
type MyQueue struct {
	stack []int
}

func Constructor232() MyQueue {
	return MyQueue{}
}
func (this *MyQueue) Push(x int) {
	this.stack = append(this.stack, x)
}
func (this *MyQueue) Pop() int {
	tmp := this.stack[0]
	this.stack = this.stack[1:]
	return tmp
}
func (this *MyQueue) Peek() int {
	return this.stack[0]
}
func (this *MyQueue) Empty() bool {
	return len(this.stack) == 0
}

/*
233. 数字 1 的个数
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了19.12%的用户
通过测试用例：38 / 38
*/
func countDigitOne(n int) int {
	res := 0
	//mulk表示10^k
	for k, mulk := 0, 1; n >= mulk; k++ {
		//每一位1的个数
		//大于10^k部分+小于等于10^k部分
		res += (n/(mulk*10))*mulk + min(max(n%(mulk*10)-mulk+1, 0), mulk)
		mulk *= 10
	}
	return res
}

/*
234. 回文链表
执行用时：116 ms, 在所有 Go 提交中击败了79.77%的用户
内存消耗：10.3 MB, 在所有 Go 提交中击败了44.11%的用户
通过测试用例：86 / 86
*/
func isPalindrome234(head *ListNode) bool {
	if head == nil {
		return true
	}

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

	mid := getMid(head)

	var reverseList func(node *ListNode) *ListNode
	reverseList = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		n1 := node
		n2 := node.Next
		node.Next = nil
		for n2 != nil {
			if n2.Next == nil {
				n2.Next = n1
				return n2
			} else {
				tmp := n2.Next
				n2.Next = n1
				n1 = n2
				n2 = tmp
			}
		}
		return n2
	}

	l1, l2 := head, reverseList(mid.Next)
	for l2 != nil {
		if l1.Val != l2.Val {
			return false
		}
		l1 = l1.Next
		l2 = l2.Next
	}
	return true
}

/*
235. 二叉搜索树的最近公共祖先
执行用时：16 ms, 在所有 Go 提交中击败了81.13%的用户
内存消耗：6.9 MB, 在所有 Go 提交中击败了56.18%的用户
通过测试用例：28 / 28
*/
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	for root != nil {
		if root.Val > q.Val && root.Val > p.Val {
			root = root.Left
		} else if root.Val < p.Val && root.Val < q.Val {
			root = root.Right
		} else {
			break
		}
	}
	return root
}

/*
236. 二叉树的最近公共祖先
执行用时：4 ms, 在所有 Go 提交中击败了99.48%的用户
内存消耗：7.4 MB, 在所有 Go 提交中击败了65.99%的用户
通过测试用例：31 / 31
*/
func lowestCommonAncestor236(root, p, q *TreeNode) *TreeNode {
	if root == nil || root == p || root == q {
		return root
	}
	l := lowestCommonAncestor236(root.Left, p, q)
	r := lowestCommonAncestor236(root.Right, p, q)
	if l == nil {
		return r
	}
	if r == nil {
		return l
	}
	return root
}

/*
237. 删除链表中的节点
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.7 MB, 在所有 Go 提交中击败了54.58%的用户
通过测试用例：41 / 41
*/
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

/*
238. 除自身以外数组的乘积
遍历两次，第一次记录左侧乘积，第二次临时变量记录右侧乘积
执行用时：28 ms, 在所有 Go 提交中击败了17.24%的用户
内存消耗：7.4 MB, 在所有 Go 提交中击败了83.55%的用户
通过测试用例：22 / 22
*/
func productExceptSelf(nums []int) []int {
	res := make([]int, len(nums))
	res[0] = 1
	for i := 1; i < len(nums); i++ {
		res[i] = res[i-1] * nums[i-1]
	}
	r := 1
	for i := len(nums) - 1; i >= 0; i-- {
		res[i] = res[i] * r
		r *= nums[i]
	}
	return res
}

/*
239. 滑动窗口最大值
执行用时：232 ms, 在所有 Go 提交中击败了24.24%的用户
内存消耗：10.1 MB, 在所有 Go 提交中击败了13.27%的用户
通过测试用例：51 / 51
*/
func maxSlidingWindow(nums []int, k int) []int {
	//单调队列 todo 抄的
	q := []int{}
	push := func(i int) {
		for len(q) > 0 && nums[i] >= nums[q[len(q)-1]] {
			q = q[:len(q)-1]
		}
		q = append(q, i)
	}

	for i := 0; i < k; i++ {
		push(i)
	}

	n := len(nums)
	res := make([]int, 1, n-k+1)
	res[0] = nums[q[0]]
	for i := k; i < n; i++ {
		push(i)
		for q[0] <= i-k {
			q = q[1:]
		}
		res = append(res, nums[q[0]])
	}
	return res
}

/*
240. 搜索二维矩阵 II
Z 字形查找，右上角(0,n−1)开始
执行用时：24 ms, 在所有 Go 提交中击败了17.86%的用户
内存消耗：6.5 MB, 在所有 Go 提交中击败了13.97%的用户
通过测试用例：129 / 129
*/
func searchMatrix(matrix [][]int, target int) bool {
	r, c := len(matrix), len(matrix[0])
	i, j := 0, c-1
	for i >= 0 && i < r && j >= 0 && j < c {
		if matrix[i][j] == target {
			return true
		} else if matrix[i][j] > target {
			j--
		} else {
			i++
		}
	}
	return false
}

/*
241. 为运算表达式设计优先级
分治
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.2 MB, 在所有 Go 提交中击败了34.14%的用户
通过测试用例：25 / 25
*/
func diffWaysToCompute(expression string) []int {
	isInt, err := strconv.Atoi(expression)
	if err == nil {
		return []int{isInt}
	}
	var res []int
	for i := 0; i < len(expression); i++ {
		tmp := string(expression[i])
		if tmp == "+" || tmp == "-" || tmp == "*" {
			l := diffWaysToCompute(expression[:i])
			r := diffWaysToCompute(expression[i+1:])
			for j := 0; j < len(l); j++ {
				for k := 0; k < len(r); k++ {
					t := 0
					if tmp == "+" {
						t = l[j] + r[k]
					}
					if tmp == "-" {
						t = l[j] - r[k]
					}
					if tmp == "*" {
						t = l[j] * r[k]
					}
					res = append(res, t)
				}
			}
		}
	}
	return res
}

/*
242. 有效的字母异位词
执行用时：12 ms, 在所有 Go 提交中击败了10.28%的用户
内存消耗：2.6 MB, 在所有 Go 提交中击败了31.33%的用户
通过测试用例：37 / 37
*/
func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	tmp := make(map[byte]int)
	for i := 0; i < len(s); i++ {
		tmp[s[i]]++
	}
	for i := 0; i < len(s); i++ {
		_, ok := tmp[t[i]]
		if !ok {
			return false
		}
		tmp[t[i]]--
		if tmp[t[i]] < 0 {
			return false
		}
	}
	return true
}

/*
243-250 todo vip
*/
