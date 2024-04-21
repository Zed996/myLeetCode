package test

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"
)

/*
todo 会员：302,305,308,311,314,317,320,323,325,333,339,340,346,348
*/

/*
301. 删除无效的括号
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了87.04%的用户
通过测试用例：127 / 127
*/
func RemoveInvalidParentheses(s string) []string {
	lremove, rremove := 0, 0
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			lremove++
		} else if s[i] == ')' {
			if lremove > 0 {
				lremove--
			} else {
				rremove++
			}
		}
	}

	var valid func(str string) bool
	valid = func(str string) bool {
		l := 0
		for i := 0; i < len(str); i++ {
			if str[i] == '(' {
				l++
			} else if str[i] == ')' {
				l--
				if l < 0 {
					return false
				}
			}
		}
		return l == 0
	}

	var res []string
	var dfs func(str string, start, lr, rr int)
	dfs = func(str string, start, lr, rr int) {
		if lr == 0 && rr == 0 {
			if valid(str) {
				res = append(res, str)
			}
			return
		}

		for i := start; i < len(str); i++ {
			//判断上一个重复
			if i != start && str[i] == str[i-1] {
				continue
			}
			//判断数量是否足够
			if lr+rr > len(str)-i {
				return
			}
			//去除左括号
			if lr > 0 && str[i] == '(' {
				dfs(str[:i]+str[i+1:], i, lr-1, rr)
			}
			//去除右括号
			if rr > 0 && str[i] == ')' {
				dfs(str[:i]+str[i+1:], i, lr, rr-1)
			}
		}

	}
	dfs(s, 0, lremove, rremove)
	return res
}

/*
303. 区域和检索 - 数组不可变
前缀和
执行用时：28 ms, 在所有 Go 提交中击败了35.16%的用户
内存消耗：9.8 MB, 在所有 Go 提交中击败了5.04%的用户
通过测试用例：15 / 15
*/
type NumArray struct {
	sum []int
}

func Constructor303(nums []int) NumArray {
	var res NumArray
	if len(nums) == 0 {
		return res
	}
	res.sum = append(res.sum, nums[0])
	for i := 1; i < len(nums); i++ {
		res.sum = append(res.sum, res.sum[i-1]+nums[i])
	}
	return res
}
func (this *NumArray) SumRange(left int, right int) int {
	if left == 0 {
		return this.sum[right]
	}
	return this.sum[right] - this.sum[left-1]
}

/*
304. 二维区域和检索 - 矩阵不可变
二维前缀和
执行用时：528 ms, 在所有 Go 提交中击败了76.87%的用户
内存消耗：19.8 MB, 在所有 Go 提交中击败了5.84%的用户
通过测试用例：22 / 22
*/
type NumMatrix struct {
	sums [][]int
}

func Constructor304(matrix [][]int) NumMatrix {
	r, c := len(matrix), len(matrix[0])
	sums := make([][]int, r+1)
	sums[0] = make([]int, c+1)
	for i := 0; i < r; i++ {
		sums[i+1] = make([]int, c+1)
		for j := 0; j < c; j++ {
			sums[i+1][j+1] = sums[i+1][j] + sums[i][j+1] - sums[i][j] + matrix[i][j]
		}
	}
	return NumMatrix{sums}
}
func (this *NumMatrix) SumRegion(row1 int, col1 int, row2 int, col2 int) int {
	return this.sums[row2+1][col2+1] - this.sums[row2+1][col1] - this.sums[row1][col2+1] + this.sums[row1][col1]
}

/*
306. 累加数
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了68.00%的用户
通过测试用例：43 / 43
*/
func IsAdditiveNumber(num string) bool {
	l := len(num)
	if l < 3 {
		return false
	}
	var dfs func(index int, arr []int) bool
	dfs = func(index int, arr []int) bool {
		if index == l {
			return true
		}
		if index > l {
			return false
		}

		tmp := arr[len(arr)-1] + arr[len(arr)-2]
		i := 1
		for tmp >= 10 {
			tmp /= 10
			i++
		}
		if index+i > l {
			return false
		}
		sToI, _ := strconv.Atoi(num[index : index+i])
		if sToI != arr[len(arr)-1]+arr[len(arr)-2] {
			return false
		}
		arr = append(arr, sToI)
		return dfs(index+i, arr)
	}

	if num[0] == '0' {
		for j := 1; j <= l/2; j++ {
			n, _ := strconv.Atoi(num[1 : j+1])
			if dfs(j+1, []int{0, n}) {
				return true
			}
		}
		return false
	}

	for i := 1; i <= l/2; i++ {
		for j := 1; j <= l-i; j++ {
			if num[i] == '0' && j > 1 {
				break
			}
			if i+j > l/3*2 {
				continue
			}
			n1, _ := strconv.Atoi(num[0:i])
			n2, _ := strconv.Atoi(num[i : i+j])
			n3, _ := strconv.Atoi(num[i+j:])
			if num[i+j:i+j+1] == "0" && n1+n2 > 0 {
				continue
			}
			if n1+n2 == n3 {
				return true
			}
			if n1+n2 < n3 {
				if dfs(i+j, []int{n1, n2}) {
					return true
				}
			}
		}
	}
	return false
}

/*
307. 区域和检索 - 数组可修改
树状数组 todo 抄的
执行用时：640 ms, 在所有 Go 提交中击败了14.44%的用户
内存消耗：22.7 MB, 在所有 Go 提交中击败了38.33%的用户
通过测试用例：16 / 16
*/
type NumArray307 struct {
	tree, nums []int
}

func Constructor307(nums []int) NumArray307 {
	tree := make([]int, len(nums)+1)
	na := NumArray307{tree: tree, nums: nums}
	for i := 0; i < len(nums); i++ {
		na.add(i+1, nums[i])
	}
	return na
}
func (this *NumArray307) lowbit(index int) int {
	//找到x的二进制数的最后一个1所表示的二进制
	return index & -index
}
func (this *NumArray307) add(index int, val int) {
	for i := index; i < len(this.tree); i += this.lowbit(i) {
		this.tree[i] += val
	}
}
func (this *NumArray307) query(index int) int {
	var res int
	for i := index; i > 0; i -= this.lowbit(i) {
		res += this.tree[i]
	}
	return res
}
func (this *NumArray307) Update(index int, val int) {
	this.add(index+1, val-this.nums[index])
	this.nums[index] = val
}
func (this *NumArray307) SumRange(left int, right int) int {
	return this.query(right+1) - this.query(left)
}

/*
309. 最佳买卖股票时机含冷冻期
动态规划
dp[i][0]--持有股票收益
dp[i][1]--不持有且冷冻期收益
dp[i][2]--不持有且非冷冻期收益
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了100.00%的用户
通过测试用例：210 / 210
*/
func maxProfit309(prices []int) int {
	if len(prices) == 0 {
		return 0
	}
	/*dp := make([][3]int, len(prices))
	dp[0][0] = -prices[0]
	dp[0][1], dp[0][2] = 0, 0
	for i := 1; i < len(prices); i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][2]-prices[i])
		dp[i][1] = dp[i-1][0] + prices[i]
		dp[i][2] = max(dp[i-1][1], dp[i-1][2])
	}
	return max(dp[len(prices)-1][1], dp[len(prices)-1][2])*/
	//空间优化
	dp0, dp1, dp2 := -prices[0], 0, 0
	for i := 1; i < len(prices); i++ {
		tmp0 := max(dp0, dp2-prices[i])
		tmp1 := dp0 + prices[i]
		tmp2 := max(dp1, dp2)
		dp0, dp1, dp2 = tmp0, tmp1, tmp2
	}
	return max(dp1, dp2)
}

/*
310. 最小高度树
执行用时：84 ms, 在所有 Go 提交中击败了21.29%的用户
内存消耗：12 MB, 在所有 Go 提交中击败了11.39%的用户
通过测试用例：71 / 71
*/
func findMinHeightTrees(n int, edges [][]int) []int {
	//入度为1的点基本不会作为最终答案【除了只有两个点的情况】
	//入度为1的相连点到其他点距离比入度为1的点小1
	//答案的点最多有两个
	//每轮排除入度为1的，然后重新生成，再排查入度为1的
	var nodes []int
	in, connect := make([]int, n), map[int][]int{}
	for _, edge := range edges {
		a, b := edge[0], edge[1]
		in[a]++
		in[b]++
		connect[a] = append(connect[a], b)
		connect[b] = append(connect[b], a)
	}
	for i := 0; i < n; i++ {
		if in[i] < 2 {
			nodes = append(nodes, i)
		}
	}
	for n > 2 {
		s := len(nodes)
		n -= s
		for _, node := range nodes {
			for _, other := range connect[node] {
				in[other]--
				if in[other] == 1 {
					nodes = append(nodes, other)
				}
			}
		}
		nodes = nodes[s:]
	}
	return nodes
}

/*
312. 戳气球
执行用时：44 ms, 在所有 Go 提交中击败了43.71%的用户
内存消耗：5.4 MB, 在所有 Go 提交中击败了38.48%的用户
通过测试用例：73 / 73
*/
func MaxCoins(nums []int) int {
	//动态规划
	//dp[i][j] = max(dp[i][k]+dp[k][j]+nums[k-1]*nums[k]*nums[k+1]) i<j-1,i<k<j
	//dp[i][j] = 0 i>=j-1
	//k是i-j区间最后一个戳破的
	n := len(nums)
	arr := []int{1}
	arr = append(arr, nums...)
	arr = append(arr, 1)
	res := make([][]int, n+2)
	for i := 0; i <= n+1; i++ {
		res[i] = make([]int, n+2)
	}
	for i := n - 1; i >= 0; i-- {
		for j := i + 2; j <= n+1; j++ {
			for k := i + 1; k < j; k++ {
				sum := arr[i] * arr[k] * arr[j]
				sum += res[i][k] + res[k][j]
				res[i][j] = max(res[i][j], sum)
			}
		}
	}

	return res[0][n+1]
}

/*
313. 超级丑数
执行用时：28 ms, 在所有 Go 提交中击败了78.40%的用户
内存消耗：3.8 MB, 在所有 Go 提交中击败了85.60%的用户
*/
func NthSuperUglyNumber(n int, primes []int) int {
	res := make([]int, n)
	res[0] = 1
	l := len(primes)
	index := make([]int, l)
	for i := 0; i < l; i++ {
		index[i] = 0
	}

	for i := 1; i < n; i++ {
		tmpIndex := 0
		tmpNum := math.MaxInt
		for j := 0; j < l; j++ {
			tmp := res[index[j]] * primes[j]
			//过滤重复情况，index都+1
			if tmp == tmpNum || tmp == res[i-1] {
				index[j]++
				continue
			}
			if tmp < tmpNum && tmp > res[i-1] {
				tmpNum = tmp
				tmpIndex = j
			}
		}
		index[tmpIndex]++
		res[i] = tmpNum
	}

	return res[n-1]
}

/*
315. 计算右侧小于当前元素的个数
执行用时：144 ms, 在所有 Go 提交中击败了90.87%的用户
内存消耗：10.9 MB, 在所有 Go 提交中击败了62.24%的用户
*/
func countSmaller(nums []int) []int {
	//类似归并排序 todo 抄的
	length := len(nums)
	index, temp, tempIndex, res := make([]int, length), make([]int, length), make([]int, length), make([]int, length)

	var merge func(arr []int, l, mid, r int)
	merge = func(arr []int, l, mid, r int) {
		i := l
		j := mid + 1
		p := l
		for i <= mid && j <= r {
			if arr[i] <= arr[j] {
				temp[p] = arr[i]
				tempIndex[p] = index[i]
				res[index[i]] += j - mid - 1
				i++
				p++
			} else {
				temp[p] = arr[j]
				tempIndex[p] = index[j]
				j++
				p++
			}
		}

		for i <= mid {
			temp[p] = arr[i]
			tempIndex[p] = index[i]
			res[index[i]] += j - mid - 1
			i++
			p++
		}

		for j <= r {
			temp[p] = arr[j]
			tempIndex[p] = index[j]
			j++
			p++
		}

		for k := l; k <= r; k++ {
			index[k] = tempIndex[k]
			arr[k] = temp[k]
		}
	}

	var mergeSort func(arr []int, l, r int)
	mergeSort = func(arr []int, l, r int) {
		if l >= r {
			return
		}
		mid := (l + r) / 2
		mergeSort(arr, l, mid)
		mergeSort(arr, mid+1, r)
		merge(arr, l, mid, r)
	}

	for i := 0; i < length; i++ {
		index[i] = i
	}

	mergeSort(nums, 0, length-1)

	return res
}

/*
316. 去除重复字母
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了86.19%的用户
*/
func removeDuplicateLetters(s string) string {
	//单调栈
	left := [26]int{}
	for _, ch := range s {
		left[ch-'a']++
	}
	stack := []byte{}
	inStack := [26]bool{}
	for i := range s {
		ch := s[i]
		if !inStack[ch-'a'] {
			//当前小于栈顶元素，栈顶元素出栈
			for len(stack) > 0 && ch < stack[len(stack)-1] {
				last := stack[len(stack)-1] - 'a'
				if left[last] == 0 {
					break
				}
				stack = stack[:len(stack)-1]
				inStack[last] = false
			}
			stack = append(stack, ch)
			inStack[ch-'a'] = true
		}
		left[ch-'a']--
	}
	return string(stack)
}

/*
318. 最大单词长度乘积
执行用时：36 ms, 在所有 Go 提交中击败了22.22%的用户
内存消耗：7.5 MB, 在所有 Go 提交中击败了82.22%的用户
*/
func MaxProduct1(words []string) int {
	//位掩码计算
	masks := map[int]int{}
	for _, v := range words {
		mask := 0
		for _, ch := range v {
			mask |= 1 << (ch - 'a')
			fmt.Println("mask:", mask)
		}
		if len(v) > masks[mask] {
			masks[mask] = len(v)
		}
	}

	fmt.Println(masks)

	res := 0

	for k, v := range masks {
		for k1, v1 := range masks {
			if k&k1 == 0 && v*v1 > res {
				res = v * v1
			}
		}
	}

	return res
}

/*
319. 灯泡开关
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了16.38%的用户
*/
func bulbSwitch(n int) int {
	//对于第 k 个灯泡，它被切换的次数恰好就是 k 的约数个数。
	//如果 k 有偶数个约数，那么最终第 k 个灯泡的状态为暗；如果 k 有奇数个约数，那么最终第 k 个灯泡的状态为亮
	//k 是「完全平方数」时，它才会有奇数个约数，否则一定有偶数个约数。
	//找出 1-n 中的完全平方数的个数
	return int(math.Sqrt(float64(n) + 0.5))
}

/*
321. 拼接最大数
执行用时：12 ms, 在所有 Go 提交中击败了40.38%的用户
内存消耗：6.6 MB, 在所有 Go 提交中击败了92.31%的用户
*/
func MaxNumber(nums1 []int, nums2 []int, k int) []int {
	//单调栈,nums1取n个,nums2取k-n个，每次比大小

	var subqueue func(l int, arr []int) []int
	subqueue = func(l int, arr []int) (r []int) {
		for i := 0; i < len(arr); i++ {
			//保证剩余可选数量--len(arr)+len(r)-i-1 >= l
			for len(r) > 0 && len(arr)+len(r)-i-1 >= l && r[len(r)-1] < arr[i] {
				r = r[:len(r)-1]
			}
			if len(r) < l {
				r = append(r, arr[i])
			}
		}
		return
	}

	var less func(a1, a2 []int) bool
	less = func(a1, a2 []int) bool {
		for i := 0; i < len(a1) && i < len(a2); i++ {
			if a1[i] != a2[i] {
				return a1[i] < a2[i]
			}
		}
		return len(a1) < len(a2)
	}

	var merge func(a1, a2 []int) []int
	merge = func(a1, a2 []int) []int {
		r := make([]int, len(a1)+len(a2))
		for i := range r {
			if less(a1, a2) {
				r[i], a2 = a2[0], a2[1:]
			} else {
				r[i], a1 = a1[0], a1[1:]
			}
		}
		return r
	}

	var res []int
	index := 0
	if len(nums2) < k {
		index = k - len(nums2)
	}
	for i := index; i <= k && i <= len(nums1); i++ {
		s1 := subqueue(i, nums1)
		s2 := subqueue(k-i, nums2)
		m := merge(s1, s2)
		//fmt.Println(s1)
		//fmt.Println(s2)
		//fmt.Println(m)
		//fmt.Println()
		if less(res, m) {
			res = m
		}
	}
	return res
}

/*
322. 零钱兑换
执行用时：12 ms, 在所有 Go 提交中击败了54.92%的用户
内存消耗：6.3 MB, 在所有 Go 提交中击败了62.80%的用户
*/
func CoinChange(coins []int, amount int) int {
	if amount == 0 {
		return 0
	}
	m := make([]int, amount+1)
	for i := 1; i < amount+1; i++ {
		m[i] = amount + 1
	}
	m[0] = 0
	for i := 1; i < amount+1; i++ {
		for j := 0; j < len(coins); j++ {
			if i >= coins[j] {
				m[i] = min(m[i-coins[j]]+1, m[i])
			}
		}
	}
	if m[amount] > amount {
		return -1
	}
	return m[amount]
}

/*
324. 摆动排序 II
*/
func wiggleSort(nums []int) {
	//todo 抄的，https://leetcode.cn/problems/wiggle-sort-ii/solutions/1627858/bai-dong-pai-xu-ii-by-leetcode-solution-no0s/
	//n偶数：n−2,n−4,⋯,0,n−1,n−3,⋯,1
	//n奇数：n−1,n−3,⋯,0,n−2,n−4,⋯,1
	n := len(nums)
	x := (n + 1) / 2
	target := quickSelect(nums, x-1)

	transAddress := func(i int) int { return (2*n - 2*i - 1) % (n | 1) }
	for k, i, j := 0, 0, n-1; k <= j; k++ {
		tk := transAddress(k)
		if nums[tk] > target {
			for j > k && nums[transAddress(j)] > target {
				j--
			}
			tj := transAddress(j)
			nums[tk], nums[tj] = nums[tj], nums[tk]
			j--
		}
		if nums[tk] < target {
			ti := transAddress(i)
			nums[tk], nums[ti] = nums[ti], nums[tk]
			i++
		}
	}
}

func quickSelect(a []int, k int) int {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(a), func(i, j int) { a[i], a[j] = a[j], a[i] })
	for l, r := 0, len(a)-1; l < r; {
		pivot := a[l]
		i, j := l, r+1
		for {
			for i++; i < r && a[i] < pivot; i++ {
			}
			for j--; j > l && a[j] > pivot; j-- {
			}
			if i >= j {
				break
			}
			a[i], a[j] = a[j], a[i]
		}
		a[l], a[j] = a[j], pivot
		if j == k {
			break
		} else if j < k {
			l = j + 1
		} else {
			r = j - 1
		}
	}
	return a[k]
}

/*
326. 3 的幂
执行用时：20 ms, 在所有 Go 提交中击败了67.49%的用户
内存消耗：6 MB, 在所有 Go 提交中击败了66.26%的用户
*/
func IsPowerOfThree(n int) bool {
	if n < 1 {
		return false
	}
	if n == 1 {
		return true
	}
	for {
		if n%3 != 0 {
			return false
		} else {
			n /= 3
		}
		if n == 1 {
			return true
		}
	}
}

/*
327. 区间和的个数
执行用时：156 ms, 在所有 Go 提交中击败了59.52%的用户
内存消耗：9.9 MB, 在所有 Go 提交中击败了53.57%的用户
*/
func countRangeSum(nums []int, lower, upper int) int {
	//todo 抄的，为什么可以有序
	var mergeCount func([]int) int
	mergeCount = func(arr []int) int {
		n := len(arr)
		if n <= 1 {
			return 0
		}

		n1 := append([]int(nil), arr[:n/2]...)
		n2 := append([]int(nil), arr[n/2:]...)
		count := mergeCount(n1) + mergeCount(n2) // 递归完毕后，n1 和 n2 均为有序

		// 统计下标对的数量
		l, r := 0, 0
		for _, v := range n1 {
			for l < len(n2) && n2[l]-v < lower {
				l++
			}
			for r < len(n2) && n2[r]-v <= upper {
				r++
			}
			count += r - l
		}

		// n1 和 n2 归并填入 arr
		p1, p2 := 0, 0
		for i := range arr {
			if p1 < len(n1) && (p2 == len(n2) || n1[p1] <= n2[p2]) {
				arr[i] = n1[p1]
				p1++
			} else {
				arr[i] = n2[p2]
				p2++
			}
		}
		return count
	}

	prefixSum := make([]int, len(nums)+1)
	for i, v := range nums {
		prefixSum[i+1] = prefixSum[i] + v
	}
	return mergeCount(prefixSum)
}

/*
328. 奇偶链表
执行用时：4 ms, 在所有 Go 提交中击败了75.65%的用户
内存消耗：3.1 MB, 在所有 Go 提交中击败了58.38%的用户
*/
func OddEvenList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	if head.Next == nil {
		return head
	}

	o, e, end := head, head.Next, head.Next
	for o != nil && e != nil {
		if o.Next == nil || e.Next == nil {
			break
		}
		o.Next = e.Next
		e.Next = o.Next.Next
		o = o.Next
		e = e.Next
	}

	/*t1 := head
	for t1 != nil {
		fmt.Print(t1.Val)
		fmt.Print(" ")
		t1 = t1.Next
	}
	fmt.Println()
	t2 := end
	for t2 != nil {
		fmt.Print(t2.Val)
		fmt.Print(" ")
		t2 = t2.Next
	}*/

	o.Next = end

	/*t1 := head
	for t1 != nil {
		fmt.Print(t1.Val)
		fmt.Print(" ")
		t1 = t1.Next
	}*/

	return head
}

/*
329. 矩阵中的最长递增路径
执行用时：24 ms, 在所有 Go 提交中击败了95.39%的用户
内存消耗：6.8 MB, 在所有 Go 提交中击败了79.61%的用户
*/
func LongestIncreasingPath(matrix [][]int) int {
	if len(matrix) == 0 {
		return 0
	}
	r, c := len(matrix), len(matrix[0])
	dp := make([][]int, r)
	for i := 0; i < r; i++ {
		dp[i] = make([]int, c)
	}

	res := 0
	var dfs func(i, j int)
	dfs = func(i, j int) {
		//fmt.Println("i:", i)
		//fmt.Println("j:", j)
		//fmt.Println()
		if i < 0 || j < 0 || i >= r || j >= c {
			return
		}
		//tmp记录周围最大值
		tmp := -1
		if i > 0 && matrix[i][j] < matrix[i-1][j] {
			if dp[i-1][j] == 0 {
				dfs(i-1, j)
			}
			tmp = max(tmp, dp[i-1][j])
		}

		if i < r-1 && matrix[i][j] < matrix[i+1][j] {
			if dp[i+1][j] == 0 {
				dfs(i+1, j)
			}
			tmp = max(tmp, dp[i+1][j])
		}

		if j > 0 && matrix[i][j] < matrix[i][j-1] {
			if dp[i][j-1] == 0 {
				dfs(i, j-1)
			}
			tmp = max(tmp, dp[i][j-1])
		}

		if j < c-1 && matrix[i][j] < matrix[i][j+1] {
			if dp[i][j+1] == 0 {
				dfs(i, j+1)
			}
			tmp = max(tmp, dp[i][j+1])
		}

		if tmp > 0 {
			dp[i][j] = tmp + 1
		} else {
			dp[i][j] = 1
		}
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if dp[i][j] == 0 {
				dfs(i, j)
			}
			res = max(res, dp[i][j])
		}
	}

	//for _, v := range dp {
	//	for _, v1 := range v {
	//		fmt.Print(v1, " ")
	//	}
	//	fmt.Println()
	//}

	return res
}

/*
330. 按要求补齐数组
执行用时：4 ms, 在所有 Go 提交中击败了85.71%的用户
内存消耗：3.2 MB, 在所有 Go 提交中击败了57.14%的用户
*/
func MinPatches(nums []int, n int) int {
	//贪心
	//如果[1,x-1]内均已覆盖,且x在数组中,则[1,2x)内均以覆盖
	//如果a添加入nums,a<=x时[1,x+a)完全覆盖；a>x时[1,x+a)未完全覆盖需要补充
	index, edge, res := 0, 1, 0
	for edge <= n {
		if index < len(nums) && nums[index] <= edge {
			edge += nums[index]
			index++
		} else {
			edge *= 2
			res++
		}
	}
	return res
}

/*
331. 验证二叉树的前序序列化
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.6 MB, 在所有 Go 提交中击败了58.33%的用户
*/
func IsValidSerialization(preorder string) bool {
	//方法1:x # # 替换为 #

	//方法2:入度=出度
	//r记录(出度-入度)的值,初始化为1因为 root为0入度2出度,加入节点时入度-1出度+2
	arr := strings.Split(preorder, ",")
	r := 1
	for _, v := range arr {
		r--
		if r < 0 {
			return false
		}
		if v != "#" {
			r += 2
		}
	}
	return r == 0
}

/*
332. 重新安排行程
执行用时：8 ms, 在所有 Go 提交中击败了93.47%的用户
内存消耗：5.8 MB, 在所有 Go 提交中击败了82.45%的用户
*/
func FindItinerary(tickets [][]string) []string {
	//Hierholzer 算法用于在连通图中寻找欧拉路径，其流程如下：
	//
	//从起点出发，进行深度优先搜索。
	//每次沿着某条边从某个顶点移动到另外一个顶点的时候，都需要删除这条边。
	//如果没有可移动的路径，则将所在节点加入到栈中，并返回。
	var (
		m   = map[string][]string{}
		res []string
	)

	for _, ticket := range tickets {
		src, dst := ticket[0], ticket[1]
		m[src] = append(m[src], dst)
	}
	for key := range m {
		sort.Strings(m[key])
	}

	var dfs func(curr string)
	dfs = func(curr string) {
		for {
			if v, ok := m[curr]; !ok || len(v) == 0 {
				break
			}
			tmp := m[curr][0]
			m[curr] = m[curr][1:]
			dfs(tmp)
		}
		res = append(res, curr)
	}

	dfs("JFK")
	for i := 0; i < len(res)/2; i++ {
		res[i], res[len(res)-1-i] = res[len(res)-1-i], res[i]
	}
	return res
}

/*
334. 递增的三元子序列
执行用时：112 ms, 在所有 Go 提交中击败了98.31%的用户
内存消耗：20.6 MB, 在所有 Go 提交中击败了73.37%的用户
*/
func IncreasingTriplet(nums []int) bool {
	l := len(nums)
	if l < 3 {
		return false
	}
	//记录三元组前两个元素，尽可能小
	first, second := nums[0], math.MaxInt
	for i := 1; i < l; i++ {
		tmp := nums[i]
		if tmp > second {
			return true
		} else if tmp > first {
			second = tmp
		} else {
			first = tmp
		}
	}
	return false
}

/*
335. 路径交叉
执行用时：12 ms, 在所有 Go 提交中击败了88.89%的用户
内存消耗：6.95 MB, 在所有 Go 提交中击败了100.00%的用户
*/
func IsSelfCrossing(distance []int) bool {
	//三种情况
	l := len(distance)
	if l < 4 {
		return false
	}
	for i := 3; i < l; i++ {
		//1和4相交
		if distance[i] >= distance[i-2] && distance[i-1] <= distance[i-3] {
			return true
		}
		//1和5相交
		if i > 3 && distance[i]+distance[i-4] >= distance[i-2] && distance[i-3] == distance[i-1] {
			return true
		}
		//1和6相交
		if i > 4 && distance[i-3] >= distance[i-1] && distance[i-2] > distance[i-4] &&
			distance[i]+distance[i-4] >= distance[i-2] &&
			distance[i-1]+distance[i-5] >= distance[i-3] {
			return true
		}
	}
	return false
}

/*
336.回文对
todo 抄的
执行用时：248 ms, 在所有 Go 提交中击败了87.50%的用户
内存消耗：13.03 MB, 在所有 Go 提交中击败了100.00%的用户
*/
func palindromePairs(words []string) [][]int {
	//words = ["abcd","dcba","lls","s","sssll"]
	//[[0,1],[1,0],[3,2],[2,4]]
	//可拼接成的回文串为 ["dcbaabcd","abcddcba","slls","llssssll"]

	//双前缀树
	//todo

	//串转hash
	var validPalind func(s string) bool
	validPalind = func(s string) bool {
		for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
			if s[i] != s[j] {
				return false
			}
		}
		return true
	}

	var reverse func(s string) string

	reverse = func(s string) string {
		res := make([]byte, len(s))
		for i, j := 0, len(s)-1; i <= j; i, j = i+1, j-1 {
			res[i], res[j] = s[j], s[i]
		}
		return string(res)
	}

	wordRev := make([]string, len(words))
	for i, v := range words {
		wordRev[i] = reverse(v)
	}
	hash := map[string]int{}
	for i, v := range wordRev {
		hash[v] = i
	}
	ans := [][]int{}
	for index, word := range words {
		for i, j := 0, len(word); i <= len(word) && j >= 0; i, j = i+1, j-1 {
			//i!=0或j!=n必选其一，否则会重复，因为word[:0] == word[n:n] == ""
			if i != 0 && validPalind(word[:i]) {
				if v, ok := hash[word[i:]]; ok && v != index {
					ans = append(ans, []int{v, index})
				}
			}
			if validPalind(word[j:len(word)]) {
				if v, ok := hash[word[:j]]; ok && v != index {
					ans = append(ans, []int{index, v})
				}
			}
		}
	}
	return ans
}

/*
337. 打家劫舍 III
执行用时：4 ms, 在所有 Go 提交中击败了91.26%的用户
内存消耗：4.89 MB, 在所有 Go 提交中击败了60.58%的用户
*/
func Rob337(root *TreeNode) int {
	//当前节点选择不偷：当前节点能偷到的最大钱数 = 左孩子能偷到的钱 + 右孩子能偷到的钱
	//当前节点选择偷：当前节点能偷到的最大钱数 = 左孩子选择自己不偷时能得到的钱 + 右孩子选择不偷时能得到的钱 + 当前节点的钱数

	var nodeMax func(node *TreeNode) [2]int
	nodeMax = func(node *TreeNode) [2]int {
		res := [2]int{0, 0}
		if node == nil {
			return res
		}
		l := nodeMax(node.Left)
		r := nodeMax(node.Right)
		res[0] = max(l[0], l[1]) + max(r[0], r[1])
		res[1] = l[0] + r[0] + node.Val
		return res
	}

	r := nodeMax(root)
	return max(r[0], r[1])
}

/*
338. 比特位计数
执行用时：4 ms, 在所有 Go 提交中击败了78.70%的用户
内存消耗：4.32 MB, 在所有 Go 提交中击败了47.12%的用户
*/
func CountBits(n int) []int {
	//i&(i-1): 将i的二进制表示中的最低位为1的改为0
	res := make([]int, n+1)
	for i := 1; i <= n; i++ {
		fmt.Println(i & (i - 1))
		res[i] = res[i&(i-1)] + 1
	}
	return res
}

/*
341. 扁平化嵌套列表迭代器
todo
*/

/*
342. 4的幂
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.98 MB, 在所有 Go 提交中击败了42.19%的用户
*/
func isPowerOfFour(n int) bool {
	//2的幂且模3余1
	return n > 0 && n&(n-1) == 0 && n%3 == 1
}

/*
343. 整数拆分
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.79 MB, 在所有 Go 提交中击败了96.02%的用户
*/
func IntegerBreak(n int) int {
	//尽可能多的 3
	if n < 4 {
		return n - 1
	}
	quotient := n / 3
	remainder := n % 3
	if remainder == 0 {
		return int(math.Pow(3, float64(quotient)))
	} else if remainder == 1 {
		return int(math.Pow(3, float64(quotient-1))) * 4
	}
	return int(math.Pow(3, float64(quotient))) * 2
}

/*
344. 反转字符串
执行用时：28 ms, 在所有 Go 提交中击败了79.54%的用户
内存消耗：6.25 MB, 在所有 Go 提交中击败了92.76%的用户
*/
func reverseString344(s []byte) {
	i, j := 0, len(s)-1
	for i < j {
		s[i], s[j] = s[j], s[i]
		i++
		j--
	}
	return
}

/*
345. 反转字符串中的元音字母
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：3.92 MB, 在所有 Go 提交中击败了48.35%的用户
*/
func reverseVowels(s string) string {
	arr := []byte(s)
	i, j := 0, len(arr)-1
	var in func(b byte) bool
	in = func(b byte) bool {
		if b == 'a' || b == 'e' || b == 'i' || b == 'o' || b == 'u' || b == 'A' || b == 'E' || b == 'I' || b == 'O' || b == 'U' {
			return true
		}
		return false
	}
	for i < j {
		if !in(arr[i]) {
			i++
		}
		if !in(arr[j]) {
			j--
		}
		if in(arr[i]) && in(arr[j]) {
			arr[i], arr[j] = arr[j], arr[i]
			i++
			j--
		}
	}
	return string(arr)
}

/*
347. 前 K 个高频元素
执行用时：12 ms, 在所有 Go 提交中击败了62.66%的用户
内存消耗：5.89 MB, 在所有 Go 提交中击败了28.51%的用户
*/
func TopKFrequent(nums []int, k int) []int {
	m := make(map[int]int)
	for _, v := range nums {
		m[v]++
	}
	var arr [][2]int
	for k, v := range m {
		arr = append(arr, [2]int{k, v})
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i][1] > arr[j][1]
	})
	var res []int
	for i := 0; i < k; i++ {
		res = append(res, arr[i][0])
	}
	return res
}

/*
349. 两个数组的交集
执行用时：3 ms, 在所有 Go 提交中击败了53.00%的用户
内存消耗：2.74 MB, 在所有 Go 提交中击败了94.50%的用户
*/
func intersection(nums1 []int, nums2 []int) []int {
	sort.Ints(nums1)
	sort.Ints(nums2)
	l1, l2 := len(nums1), len(nums2)
	var res []int
	for i, j := 0, 0; i < l1 && j < l2; {
		if nums1[i] == nums2[j] {
			if len(res) == 0 {
				res = append(res, nums1[i])
			} else if nums1[i] > res[len(res)-1] {
				res = append(res, nums1[i])
			}
			i++
			j++
		} else if nums1[i] > nums2[j] {
			j++
		} else {
			i++
		}
	}

	return res
}

/*
350. 两个数组的交集 II
执行用时：3 ms, 在所有 Go 提交中击败了59.65%的用户
内存消耗：2.48 MB, 在所有 Go 提交中击败了92.80%的用户
*/
func intersect(nums1 []int, nums2 []int) []int {
	sort.Ints(nums1)
	sort.Ints(nums2)
	l1, l2 := len(nums1), len(nums2)
	var res []int
	for i, j := 0, 0; i < l1 && j < l2; {
		if nums1[i] == nums2[j] {
			res = append(res, nums1[i])
			i++
			j++
		} else if nums1[i] > nums2[j] {
			j++
		} else {
			i++
		}
	}

	return res
}
