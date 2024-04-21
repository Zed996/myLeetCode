package test

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
)

/*
51. N 皇后
执行用时：16 ms, 在所有 Go 提交中击败了6.86%的用户
内存消耗：3.8 MB, 在所有 Go 提交中击败了10.37%的用户
通过测试用例：9 / 9
*/
func SolveNQueens(n int) [][]string {
	var res [][]string
	var s []string
	dfs51(0, n, s, &res)
	return res
}
func valid51(i, n int, s []string) []int {
	if i >= n || i < 0 {
		return []int{}
	}
	var res []int
	for j := 0; j < n; j++ {
		q := 0
		//判断上方有没有
		for k := 0; k < i; k++ {
			if string(s[k][j]) == "Q" {
				q++
			}
		}
		//判断左上
		c := j - 1
		r := i - 1
		for c >= 0 && r >= 0 {
			if string(s[r][c]) == "Q" {
				q++
			}
			c--
			r--
		}
		//判断右上
		c = j + 1
		r = i - 1
		for c < n && r >= 0 {
			if string(s[r][c]) == "Q" {
				q++
			}
			c++
			r--
		}
		if q == 0 {
			res = append(res, j)
		}
	}
	return res
}
func dfs51(i int, n int, s []string, res *[][]string) bool {

	if i == n {
		*res = append(*res, append([]string{}, s...))
		return true
	}

	valid := valid51(i, n, s)
	if len(valid) == 0 {
		return false
	}

	for j := 0; j < len(valid); j++ {
		tmp := ""
		for k := 0; k < n; k++ {
			if k == valid[j] {
				tmp += "Q"
			} else {
				tmp += "."
			}
		}
		s = append(s, tmp)
		dfs51(i+1, n, s, res)
		s = s[:len(s)-1]
	}
	return false
}

/*
52. N皇后 II
执行用时：12 ms, 在所有 Go 提交中击败了17.03%的用户
内存消耗：2.7 MB, 在所有 Go 提交中击败了5.77%的用户
通过测试用例：9 / 9
*/
func TotalNQueens(n int) int {
	var res int
	var s []string
	dfs52(0, n, s, &res)
	return res
}
func dfs52(i int, n int, s []string, res *int) bool {

	if i == n {
		*res++
		return true
	}

	valid := valid51(i, n, s)
	if len(valid) == 0 {
		return false
	}

	for j := 0; j < len(valid); j++ {
		tmp := ""
		for k := 0; k < n; k++ {
			if k == valid[j] {
				tmp += "Q"
			} else {
				tmp += "."
			}
		}
		s = append(s, tmp)
		dfs52(i+1, n, s, res)
		s = s[:len(s)-1]
	}
	return false
}

/*
53. 最大子数组和
当“连续和”为负数的时候就应该从下一个元素重新计算“连续和”，因为负数加上下一个元素 “连续和”只会变小不会变大
执行用时：80 ms, 在所有 Go 提交中击败了90.71%的用户
内存消耗：9.3 MB, 在所有 Go 提交中击败了45.09%的用户
通过测试用例：209 / 209
*/
func MaxSubArray(nums []int) int {
	max := nums[0]
	tmp := 0
	n := len(nums)
	for i := 0; i < n; i++ {
		tmp += nums[i]
		if tmp > max {
			max = tmp
		}
		if tmp < 0 {
			tmp = 0
		}
	}
	return max
}

/*
54. 螺旋矩阵
方法1.打印第一行，逆时针旋转90°
方法2.由外至内减少层数
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了16.63%的用户
通过测试用例：23 / 23
*/
func SpiralOrder(matrix [][]int) []int {
	var res []int
	iStart := 0
	iEnd := len(matrix) - 1
	jStart := 0
	jEnd := len(matrix[0]) - 1
	for iStart <= iEnd && jStart <= jEnd {
		//fmt.Println("iStart:", iStart)
		//fmt.Println("iEnd:", iEnd)
		//fmt.Println("jStart:", jStart)
		//fmt.Println("jEnd:", jEnd)
		if iStart == iEnd {
			for j := jStart; j <= jEnd; j++ {
				res = append(res, matrix[iStart][j])
			}
			break
		}
		if jStart == jEnd {
			for i := iStart; i <= iEnd; i++ {
				res = append(res, matrix[i][jEnd])
			}
			break
		}
		for j := jStart; j <= jEnd-1; j++ {
			res = append(res, matrix[iStart][j])
		}
		for i := iStart; i <= iEnd-1; i++ {
			res = append(res, matrix[i][jEnd])
		}
		for j := jEnd; j >= jStart+1; j-- {
			res = append(res, matrix[iEnd][j])
		}
		for i := iEnd; i >= iStart+1; i-- {
			res = append(res, matrix[i][jStart])
		}
		iStart++
		iEnd--
		jStart++
		jEnd--
	}
	return res
}

/*
55. 跳跃游戏
执行用时：52 ms, 在所有 Go 提交中击败了46.56%的用户
内存消耗：6.7 MB, 在所有 Go 提交中击败了79.03%的用户
通过测试用例：170 / 170
*/
func CanJump(nums []int) bool {
	n := len(nums)
	far := 0
	r := 0
	for i := 0; i < n; i++ {
		if i+nums[i] > far {
			far = i + nums[i]
		}
		if i == r {
			if r == far && r < n-1 {
				return false
			}
			r = far
		}
	}
	return true
}

/*
56. 合并区间
执行用时：24 ms, 在所有 Go 提交中击败了27.73%的用户
内存消耗：6.7 MB, 在所有 Go 提交中击败了81.56%的用户
通过测试用例：169 / 169
*/
func Merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool { return intervals[i][0] < intervals[j][0] })
	index := 0
	var res [][]int
	res = append(res, intervals[0])
	for i := 1; i < len(intervals); i++ {
		if res[index][1] >= intervals[i][0] {
			//合并
			if intervals[i][1] > res[index][1] {
				res[index][1] = intervals[i][1]
			}
		} else {
			index++
			res = append(res, intervals[i])
		}
	}
	return res
}

/*
57. 插入区间
用上一题方法：
执行用时：8 ms, 在所有 Go 提交中击败了58.99%的用户
内存消耗：4.5 MB, 在所有 Go 提交中击败了23.25%的用户
通过测试用例：156 / 156
每个判断:
执行用时：12 ms, 在所有 Go 提交中击败了9.43%的用户
内存消耗：4.5 MB, 在所有 Go 提交中击败了60.31%的用户
通过测试用例：156 / 156
*/
func insert(intervals [][]int, newInterval []int) [][]int {
	/*
		intervals = append(intervals, newInterval)
			sort.Slice(intervals, func(i, j int) bool { return intervals[i][0] < intervals[j][0] })
			index := 0
			var res [][]int
			res = append(res, intervals[0])
			for i := 1; i < len(intervals); i++ {
				if res[index][1] >= intervals[i][0] {
					//合并
					if intervals[i][1] > res[index][1] {
						res[index][1] = intervals[i][1]
					}
				} else {
					index++
					res = append(res, intervals[i])
				}
			}
			return res
	*/
	l := newInterval[0]
	r := newInterval[1]
	merge := false
	var res [][]int
	for _, v := range intervals {
		if l > v[1] {
			res = append(res, v)
		} else if r < v[0] {
			if !merge {
				res = append(res, []int{l, r})
			}
			merge = true
			res = append(res, v)
		} else {
			if v[0] < l {
				l = v[0]
			}
			if v[1] > r {
				r = v[1]
			}
		}
	}
	if !merge {
		res = append(res, []int{l, r})
	}
	return res
}

/*
58. 最后一个单词的长度
使用go自带库函数
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了21.89%的用户
通过测试用例：58 / 58
手写
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了100.00%的用户
通过测试用例：58 / 58
*/
func LengthOfLastWord(s string) int {
	//tmp := strings.Fields(s)
	//return len(tmp[len(tmp)-1])
	num := 0
	start := false
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] != ' ' {
			start = true
			num++
		}
		if start && s[i] == ' ' {
			return num
		}
	}
	return num
}

/*
59. 螺旋矩阵 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了96.75%的用户
通过测试用例：20 / 20
*/
func GenerateMatrix(n int) [][]int {
	//分(n+1)/2层，每层4n-4个，n-=2
	var res [][]int
	for i := 0; i < n; i++ {
		t := make([]int, n)
		res = append(res, t)
	}
	//num代表1至n*n
	num := 1
	//l代表层数
	l := (n + 1) / 2
	for k := 0; k < l; k++ {
		is := k
		ie := n - k - 1
		js := k
		je := n - k - 1
		//fmt.Println("is:", is)
		//fmt.Println("ie:", ie)
		//fmt.Println("js:", js)
		//fmt.Println("je:", je)
		if is >= ie {
			//fmt.Println("num")
			res[is][js] = num
			return res
		}
		for j := js; j <= je-1; j++ {
			res[is][j] = num
			num++
		}
		for i := is; i <= ie-1; i++ {
			res[i][je] = num
			num++
		}
		for j := je; j >= js+1; j-- {
			res[ie][j] = num
			num++
		}
		for i := ie; i >= is+1; i-- {
			res[i][js] = num
			num++
		}
	}
	return res
}

/*
60. 排列序列
因为全排列。按顺序排。开头肯定也是1开头的有多少个2开头的也多少个3开头的 这样 1后面的是n-1的阶乘个
然后我们通过k和n-1阶乘之间的关系。来逐个的从头到尾的判断出每个位置的数字
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了56.65%的用户
通过测试用例：200 / 200
*/
func GetPermutation(n int, k int) string {
	factorial := make([]int, n)
	factorial[0] = 1
	for i := 1; i < n; i++ {
		factorial[i] = factorial[i-1] * i
	}
	k--
	res := ""
	//valid记录每一个元素是否被使用过
	valid := make([]int, n+1)
	for i := 0; i < len(valid); i++ {
		valid[i] = 1
	}
	for i := 1; i <= n; i++ {
		order := k/factorial[n-i] + 1
		fmt.Println("order:", order)
		for j := 1; j <= n; j++ {
			order -= valid[j]
			if order == 0 {
				res += strconv.Itoa(j)
				valid[j] = 0
				break
			}
		}
		fmt.Println("valid:", valid)
		k %= factorial[n-i]
	}
	return res
}

/*
61. 旋转链表
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了100.00%的用户
通过测试用例：231 / 231
*/
func RotateRight(head *ListNode, k int) *ListNode {
	if head == nil {
		return head
	}
	zero := &ListNode{
		Val:  0,
		Next: head,
	}
	if k == 0 {
		return head
	}
	//原本链表转为环形链表
	l := 1
	round := head
	for round.Next != nil {
		round = round.Next
		l++
	}
	round.Next = head

	tmp := zero
	for i := 0; i < l-k%l; i++ {
		tmp = tmp.Next
	}
	zero.Next = tmp.Next
	tmp.Next = nil
	return zero.Next
}

/*
62. 不同路径
移动 m+n−2 次，其中有m−1 次向下移动，n−1 次向右移动
 m+n−2 次移动中选择 m−1 次向下移动的方案数
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了86.97%的用户
通过测试用例：63 / 63
*/
func UniquePaths(m int, n int) int {
	return int(new(big.Int).Binomial(int64(m+n-2), int64(n-1)).Int64())
}
func Factorial62(x float64, y float64) float64 {
	//从x乘到y
	res := 1.0
	for i := x; i <= y; i++ {
		res *= i
	}
	return res
}

/*
63. 不同路径 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了6.12%的用户
通过测试用例：41 / 41
*/
func UniquePathsWithObstacles(obstacleGrid [][]int) int {
	if obstacleGrid[0][0] == 1 {
		return 0
	}
	r := len(obstacleGrid)
	c := len(obstacleGrid[0])
	tmp := [][]int{}
	for i := 0; i < r; i++ {
		tmp = append(tmp, make([]int, c))
	}
	//fmt.Println(tmp)
	tmp[0][0] = 1
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if obstacleGrid[i][j] == 1 {
				tmp[i][j] = 0
			} else if i == 0 {
				if j > 0 {
					tmp[i][j] = tmp[i][j-1]
				}
			} else if j == 0 {
				if i > 0 {
					tmp[i][j] = tmp[i-1][j]
				}
			} else {
				tmp[i][j] = tmp[i-1][j] + tmp[i][j-1]
			}
		}
	}
	fmt.Println(tmp)
	return tmp[r-1][c-1]
}

/*
64. 最小路径和
执行用时：4 ms, 在所有 Go 提交中击败了98.17%的用户
内存消耗：4.2 MB, 在所有 Go 提交中击败了8.78%的用户
通过测试用例：61 / 61
*/
func MinPathSum(grid [][]int) int {
	r := len(grid)
	c := len(grid[0])
	tmp := [][]int{}
	for i := 0; i < r; i++ {
		tmp = append(tmp, make([]int, c))
	}
	//fmt.Println(tmp)
	tmp[0][0] = grid[0][0]
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if i == 0 {
				if j > 0 {
					tmp[i][j] = tmp[i][j-1] + grid[i][j]
				}
			} else if j == 0 {
				if i > 0 {
					tmp[i][j] = tmp[i-1][j] + grid[i][j]
				}
			} else {
				min := tmp[i-1][j]
				if tmp[i][j-1] < min {
					min = tmp[i][j-1]
				}
				tmp[i][j] = min + grid[i][j]
			}
		}
	}
	fmt.Println(tmp)
	return tmp[r-1][c-1]
}

/*
65. 有效数字
有限状态自动机
1：开始，可达状态-234
2：数字，可达状态-2456,0-9 ascII 48-57
3：正负号，可达状态-2，+ 43，- 45
4：小数点，可达状态-256，. 46
5：eE，可达状态-23，e 101，E 69
6：结束--return true
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.3 MB, 在所有 Go 提交中击败了30.16%的用户
通过测试用例：1490 / 1490
*/
func IsNumber(s string) bool {
	m := make(map[int][]int)
	m[1] = []int{2, 3, 4}
	m[2] = []int{2, 4, 5, 6}
	m[3] = []int{2, 4}
	m[4] = []int{2, 6}
	m[5] = []int{2, 3}
	l := len(s)
	if l > 1 && s[0:2] == ".e" {
		return false
	}
	if getStatus65(s[0]) == 5 {
		return false
	}
	if getStatus65(s[l-1]) != 2 && getStatus65(s[l-1]) != 4 {
		return false
	}
	eAppeared := false
	statusNum := make(map[int]int)
	status := 1
	for i := 0; i < l; i++ {
		status = getStatus65(s[i])
		if status == 5 {
			eAppeared = true
		}
		//e后禁止小数点
		if eAppeared && status == 4 {
			return false
		}
		statusNum[status]++
		fmt.Println("i:", i)
		fmt.Println("status:", status)
		fmt.Println("statusNum:", statusNum)
		//e后加减号，跳一次
		if status == 5 && i < l-1 && getStatus65(s[i+1]) == 3 {
			i++
			continue
		}
		if status != 2 && statusNum[status] > 1 {
			return false
		}
		if i < l-1 {
			nextStatus := getStatus65(s[i+1])
			fmt.Println("nextStatus:", nextStatus)
			if !intInArrInts(nextStatus, m[status]) {
				return false
			}
		}
	}
	if statusNum[2] == 0 {
		return false
	}
	return true
}
func intInArrInts(i int, arr []int) bool {
	for _, v := range arr {
		if v == i {
			return true
		}
	}
	return false
}
func getStatus65(i uint8) int {
	if i >= 48 && i <= 57 {
		return 2
	}
	if i == 43 || i == 45 {
		return 3
	}
	if i == 46 {
		return 4
	}
	if i == 101 || i == 69 {
		return 5
	}
	return 0
}

/*
66. 加一
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了96.12%的用户
通过测试用例：111 / 111
*/
func PlusOne(digits []int) []int {
	l := len(digits)
	digits[l-1]++
	add := 0
	for i := l - 1; i >= 0; i-- {
		digits[i] += add
		if digits[i] > 9 {
			digits[i] = digits[i] - 10
			add = 1
		} else {
			add = 0
		}
		if add == 0 {
			break
		}
	}
	if add > 0 {
		res := []int{1}
		res = append(res, digits...)
		return res
	}
	return digits
}

/*
67. 二进制求和
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了79.30%的用户
通过测试用例：294 / 294
*/
func AddBinary(a string, b string) string {
	l1 := len(a)
	l2 := len(b)
	if l1 > l2 {
		return AddBinary(b, a)
	}
	tmpZero := ""
	for i := 0; i < l2-l1; i++ {
		tmpZero += "0"
	}
	a = tmpZero + a
	fmt.Println("a:", a)
	fmt.Println("b:", b)
	fmt.Println(l2)
	r := []byte(b)
	add := false
	for i := l2 - 1; i >= 0; i-- {
		if r[i] == '0' {
			if a[i] == '0' {
				if add {
					r[i] = '1'
					add = false
				}
			} else {
				if !add {
					r[i] = '1'
				}
			}
		} else {
			if a[i] == '0' {
				if add {
					r[i] = '0'
				}
			} else {
				if !add {
					r[i] = '0'
					add = true
				}
			}
		}
	}
	fmt.Println("r:", r)
	fmt.Println("add:", add)
	if add {
		t := []byte{'1'}
		t = append(t, r...)
		return string(t)
	}
	return string(r)
}

/*
68. 文本左右对齐
抄的答案
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了48.00%的用户
通过测试用例：27 / 27
*/
func fullJustify(words []string, maxWidth int) []string {
	var res []string
	right, n := 0, len(words)
	for {
		left := right // 当前行的第一个单词在 words 的位置
		sumLen := 0   // 统计这一行单词长度之和
		// 循环确定当前行可以放多少单词，注意单词之间应至少有一个空格
		for right < n && sumLen+len(words[right])+right-left <= maxWidth {
			sumLen += len(words[right])
			right++
		}

		// 当前行是最后一行：单词左对齐，且单词之间应只有一个空格，在行末填充剩余空格
		if right == n {
			s := strings.Join(words[left:], " ")
			res = append(res, s+blank(maxWidth-len(s)))
			return res
		}

		numWords := right - left
		numSpaces := maxWidth - sumLen

		// 当前行只有一个单词：该单词左对齐，在行末填充剩余空格
		if numWords == 1 {
			res = append(res, words[left]+blank(numSpaces))
			continue
		}

		// 当前行不只一个单词
		avgSpaces := numSpaces / (numWords - 1)
		extraSpaces := numSpaces % (numWords - 1)
		s1 := strings.Join(words[left:left+extraSpaces+1], blank(avgSpaces+1)) // 拼接额外加一个空格的单词
		s2 := strings.Join(words[left+extraSpaces+1:right], blank(avgSpaces))  // 拼接其余单词
		res = append(res, s1+blank(avgSpaces)+s2)
	}
}

// blank 返回长度为 n 的由空格组成的字符串
func blank(n int) string {
	return strings.Repeat(" ", n)
}

/*
69. x 的平方根
牛顿迭代法
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了62.23%的用户
通过测试用例：1017 / 1017
*/
func mySqrt(x int) int {
	if x == 0 {
		return 0
	}
	t := float64(x)
	c := float64(x)
	for {
		t1 := (t + c/t) / 2
		if math.Abs(t1-t) < 1e-7 {
			break
		}
		t = t1
	}
	return int(t)
}

/*
70. 爬楼梯
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了78.00%的用户
通过测试用例：45 / 45
*/
func ClimbStairs(n int) int {
	if n == 1 {
		return 1
	}
	if n == 2 {
		return 2
	}
	arr := make([]int, n)
	arr[0] = 1
	arr[1] = 2
	for i := 2; i < n; i++ {
		arr[i] = arr[i-1] + arr[i-2]
	}
	fmt.Println(arr)
	return arr[n-1]
}

/*
71. 简化路径
path 根据 / 分割
空字符串或一个点-跳过，目录名-入栈，两个点-出栈
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：4 MB, 在所有 Go 提交中击败了18.37%的用户
通过测试用例：257 / 257
*/
func SimplifyPath(path string) string {
	sArr := []string{}
	tmp := strings.Split(path, "/")
	fmt.Println(tmp)
	for _, v := range tmp {
		if v == "" || v == "." {
			continue
		} else if v == ".." {
			if len(sArr) > 0 {
				sArr = sArr[:len(sArr)-1]
			}
		} else {
			sArr = append(sArr, v)
		}
	}
	fmt.Println(sArr)
	s := ""
	for _, v := range sArr {
		s += "/" + v
	}
	if len(s) == 0 {
		return "/"
	}
	return s
}

/*
72. 编辑距离
动态规划，
执行用时：8 ms, 在所有 Go 提交中击败了17.28%的用户
内存消耗：5.4 MB, 在所有 Go 提交中击败了91.74%的用户
通过测试用例：1146 / 1146
*/
func MinDistance(word1 string, word2 string) int {
	l1 := len(word1)
	l2 := len(word2)

	if l1 == 0 || l2 == 0 {
		return l1 + l2
	}

	m := make([][]int, l1+1)
	for i := 0; i <= l1; i++ {
		m[i] = make([]int, l2+1)
	}

	for i := 0; i <= l1; i++ {
		m[i][0] = i
	}
	for j := 0; j <= l2; j++ {
		m[0][j] = j
	}

	for i := 1; i <= l1; i++ {
		for j := 1; j <= l2; j++ {
			//增
			left := m[i-1][j] + 1
			//删
			down := m[i][j-1] + 1
			//改
			l_d := m[i-1][j-1]
			if word1[i-1] != word2[j-1] {
				//结尾不同则加1
				l_d++
			}
			m[i][j] = int(math.Min(float64(left), math.Min(float64(down), float64(l_d))))
		}
	}
	return m[l1][l2]

}

/*+
73. 矩阵置零
O(1空间)，两个常量记录第一行和第一列是否为0，然后遍历过程，行有0则该行1列变0，列有0则该列1行变0，最后根据常量标记和0置0
执行用时：12 ms, 在所有 Go 提交中击败了69.40%的用户
内存消耗：6.2 MB, 在所有 Go 提交中击败了5.39%的用户
通过测试用例：164 / 164
*/
func SetZeroes(matrix [][]int) {
	r := len(matrix)
	c := len(matrix[0])
	//，两个常量记录第一行和第一列是否为0
	rZero := false
	cZero := false
	for i := 0; i < r; i++ {
		if matrix[i][0] == 0 {
			rZero = true
			break
		}
	}
	for j := 0; j < c; j++ {
		if matrix[0][j] == 0 {
			cZero = true
			break
		}
	}

	//遍历过程，行有0则该行1列变0，列有0则该列1行变0
	for i := 1; i < r; i++ {
		for j := 1; j < c; j++ {
			if matrix[i][j] == 0 {
				matrix[i][0] = 0
				matrix[0][j] = 0
			}
		}
	}

	//根据0置0
	for i := 1; i < r; i++ {
		for j := 1; j < c; j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
	}

	//根据标记置0
	if rZero {
		for i := 0; i < r; i++ {
			matrix[i][0] = 0
		}
	}
	if cZero {
		for j := 0; j < c; j++ {
			matrix[0][j] = 0
		}
	}
	return
}

/*
74. 搜索二维矩阵
矩阵有序，横竖分别二分
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.5 MB, 在所有 Go 提交中击败了99.92%的用户
通过测试用例：133 / 133
*/
func SearchMatrix(matrix [][]int, target int) bool {
	r := len(matrix)
	c := len(matrix[0])
	left := 0
	right := r
	mr := (left + right) / 2
	for {
		mr = (left + right) / 2
		fmt.Println("mr:", mr)
		fmt.Println("matrix[mr/2][c-1]:", matrix[mr][c-1])
		fmt.Println("matrix[mr/2][0]:", matrix[mr][0])
		if mr > 0 && target > matrix[mr-1][c-1] && target < matrix[mr][0] {
			return false
		}
		if target > matrix[mr][c-1] {
			if mr == r-1 {
				break
			}
			left = (left + right) / 2
		} else if target < matrix[mr][0] {
			if mr == 0 {
				return false
			}
			right = (left + right + 1) / 2
		} else {
			break
		}
	}
	fmt.Println("mr:", mr)
	left = 0
	right = c
	mc := (left + right) / 2
	for {
		mc = (left + right) / 2
		fmt.Println("mc:", mc)
		fmt.Println("matrix[mr][mc-1]:", matrix[mr][mc-1])
		fmt.Println("matrix[mr][mc]:", matrix[mr][mc])
		if mc > 0 && target > matrix[mr][mc-1] && target < matrix[mr][mc] {
			return false
		}
		if target > matrix[mr][mc] {
			if mc == c-1 {
				if target == matrix[mr][mc] {
					return true
				} else {
					return false
				}
			}
			left = (left + right) / 2
		} else if target < matrix[mr][mc] {
			if mc == 0 {
				return false
			}
			right = (left + right + 1) / 2
		} else {
			return true
		}
	}
}

/*
75. 颜色分类
双指针，一个0一个2，0++，2--，2需要额外判断
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了63.00%的用户
通过测试用例：87 / 87
*/
func SortColors(nums []int) {
	l := len(nums)
	p0 := 0
	p2 := l - 1
	for i := 0; i < l; i++ {
		switch nums[i] {
		case 0:
			nums[i] = nums[p0]
			nums[p0] = 0
			p0++
		case 2:
			if p2+1 == i {
				return
			}
			nums[i] = nums[p2]
			nums[p2] = 2
			p2--
			//额外判断nums[i]
			i--
		default:
		}
	}
	return
}

/*
76. 最小覆盖子串
滑动窗口法
执行用时：108 ms, 在所有 Go 提交中击败了28.46%的用户
内存消耗：2.7 MB, 在所有 Go 提交中击败了53.66%的用户
通过测试用例：266 / 266
*/
func MinWindow(s string, t string) string {
	ls := len(s)
	lt := len(t)
	ori := make(map[byte]int)
	for i := 0; i < lt; i++ {
		ori[t[i]]++
	}
	ansL, ansR := -1, -1
	min := math.MaxInt32
	cnt := make(map[byte]int)
	less := func() bool {
		for k, v := range ori {
			if cnt[k] < v {
				return false
			}
		}
		return true
	}
	for l, r := 0, 0; r < ls; r++ {
		if r < ls && ori[s[r]] > 0 {
			cnt[s[r]]++
		}
		for less() && l <= r {
			if r-l+1 < min {
				min = r - l + 1
				ansL, ansR = l, l+min
			}
			if _, ok := ori[s[l]]; ok {
				cnt[s[l]] -= 1
			}
			l++
		}
	}
	if ansL == -1 {
		return ""
	}
	return s[ansL:ansR]
}

/*
77. 组合
执行用时：4 ms, 在所有 Go 提交中击败了97.10%的用户
内存消耗：6.3 MB, 在所有 Go 提交中击败了44.50%的用户
通过测试用例：27 / 27
*/
func Combine(n int, k int) [][]int {
	var res77 [][]int
	dfs77(n, k, 0, []int{}, &res77)
	return res77
}
func dfs77(n, k, index int, tmp []int, res77 *[][]int) {
	fmt.Println("k:", k)
	fmt.Println("index:", index)
	fmt.Println("tmp:", tmp)
	if k == 0 {
		*res77 = append(*res77, append([]int{}, tmp...))
		return
	}
	tmp = append(tmp, index+1)
	dfs77(n, k-1, index+1, tmp, res77)
	tmp = tmp[:len(tmp)-1]
	if index < n-k {
		dfs77(n, k, index+1, tmp, res77)
	}
}

/*
78. 子集
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了64.30%的用户
通过测试用例：10 / 10
*/
func Subsets(nums []int) [][]int {
	l := len(nums)
	var res78 [][]int
	dfs78(l, nums, 0, []int{}, &res78)
	return res78
}
func dfs78(l int, nums []int, index int, tmp []int, res78 *[][]int) {
	if index == l {
		*res78 = append(*res78, append([]int{}, tmp...))
		return
	}
	tmp = append(tmp, nums[index])
	dfs78(l, nums, index+1, tmp, res78)
	tmp = tmp[:len(tmp)-1]
	dfs78(l, nums, index+1, tmp, res78)
}

/*
79. 单词搜索
执行用时：52 ms, 在所有 Go 提交中击败了97.15%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了61.01%的用户
通过测试用例：83 / 83
*/
func Exist(board [][]byte, word string) bool {
	r := len(board)
	c := len(board[0])
	used := make([][]int, r)
	for i := 0; i < r; i++ {
		used[i] = make([]int, c)
	}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			b := dfs79(board, word, i, j, used)
			if b {
				return true
			}
		}
	}
	return false
}
func dfs79(board [][]byte, word string, i, j int, used [][]int) bool {
	//fmt.Println("i:", i)
	//fmt.Println("j:", j)
	//fmt.Println("word:", word)
	//fmt.Println("used:", used)
	r := len(board)
	c := len(board[0])
	if i < 0 || j < 0 || i >= r || j >= c {
		return false
	}

	if board[i][j] != word[0] {
		return false
	}
	used[i][j] = 1
	if len(word) == 1 {
		return true
	}

	if i > 0 && used[i-1][j] == 0 {
		b := dfs79(board, word[1:], i-1, j, used)
		if b {
			return true
		}
	}

	if j > 0 && used[i][j-1] == 0 {
		b := dfs79(board, word[1:], i, j-1, used)
		if b {
			return true
		}
	}

	if i < r-1 && used[i+1][j] == 0 {
		b := dfs79(board, word[1:], i+1, j, used)
		if b {
			return true
		}
	}

	if j < c-1 && used[i][j+1] == 0 {
		b := dfs79(board, word[1:], i, j+1, used)
		if b {
			return true
		}
	}
	used[i][j] = 0

	return false
}

/*
80. 删除有序数组中的重复项 II
执行用时：4 ms, 在所有 Go 提交中击败了83.33%的用户
内存消耗：2.9 MB, 在所有 Go 提交中击败了59.05%的用户
通过测试用例：164 / 164
*/
func RemoveDuplicates80(nums []int) int {
	var deletek func(k int) int
	deletek = func(k int) int {
		index := 0
		for _, v := range nums {
			if index < k || nums[index-k] != v {
				nums[index] = v
				index++
			}
		}
		return index
	}
	return deletek(2)
}

/*
81. 搜索旋转排序数组 II
执行用时：4 ms, 在所有 Go 提交中击败了85.71%的用户
内存消耗：3 MB, 在所有 Go 提交中击败了26.34%的用户
通过测试用例：280 / 280
*/
func search81(nums []int, target int) bool {
	for i := 0; i < len(nums); i++ {
		if nums[i] == target {
			return true
		}
	}
	return false
}

/*
82. 删除排序链表中的重复元素 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.9 MB, 在所有 Go 提交中击败了61.74%的用户
通过测试用例：166 / 166
*/
func deleteDuplicates(head *ListNode) *ListNode {
	tmp := &ListNode{
		Val:  0,
		Next: head,
	}
	res := tmp
	for tmp.Next != nil && tmp.Next.Next != nil {
		if tmp.Next.Val == tmp.Next.Next.Val {
			x := tmp.Next.Val
			for tmp.Next != nil && tmp.Next.Val == x {
				tmp.Next = tmp.Next.Next
			}
		} else {
			tmp = tmp.Next
		}
	}
	return res.Next
}

/*
83. 删除排序链表中的重复元素
执行用时：4 ms, 在所有 Go 提交中击败了81.26%的用户
内存消耗：3 MB, 在所有 Go 提交中击败了59.81%的用户
通过测试用例：166 / 166
*/
func DeleteDuplicates83(head *ListNode) *ListNode {
	tmp := &ListNode{
		Val:  -101,
		Next: head,
	}
	res := tmp
	for tmp.Next != nil {
		fmt.Println("val:", tmp.Val)
		fmt.Println("next:", tmp.Next)
		for tmp.Val == tmp.Next.Val {
			if tmp.Next.Next != nil {
				tmp.Next = tmp.Next.Next
			} else {
				tmp.Next = nil
				return res.Next
			}
		}
		tmp = tmp.Next
	}
	return res.Next
}

/*
84. 柱状图中最大的矩形
单调栈确认每个index的左右边界,todo 一次遍历确定左右边界
执行用时：88 ms, 在所有 Go 提交中击败了84.40%的用户
内存消耗：8.7 MB, 在所有 Go 提交中击败了71.43%的用户
通过测试用例：98 / 98
*/
func LargestRectangleArea(heights []int) int {
	l := len(heights)
	left := make([]int, l)
	right := make([]int, l)
	var stack []int

	for i := 0; i < l; i++ {
		for len(stack) > 0 && heights[i] <= heights[stack[len(stack)-1]] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			left[i] = -1
		} else {
			left[i] = stack[len(stack)-1]
		}
		stack = append(stack, i)
	}

	stack = []int{}
	for i := l - 1; i >= 0; i-- {
		for len(stack) > 0 && heights[i] <= heights[stack[len(stack)-1]] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			right[i] = l
		} else {
			right[i] = stack[len(stack)-1]
		}
		stack = append(stack, i)
	}

	res := 0
	for i := 0; i < l; i++ {
		if (right[i]-left[i]-1)*heights[i] > res {
			res = (right[i] - left[i] - 1) * heights[i]
		}
	}

	return res
}

/*
85. 最大矩形
参考84，每行及向上最大面积，遍历所有行
执行用时：16 ms, 在所有 Go 提交中击败了24.96%的用户
内存消耗：6.4 MB, 在所有 Go 提交中击败了22.18%的用户
通过测试用例：73 / 73
*/
func MaximalRectangle(matrix [][]byte) int {
	r := len(matrix)
	c := len(matrix[0])
	max := 0
	for i := 0; i < r; i++ {
		arr := make([]int, c)
		for j := 0; j < c; j++ {
			for k := i; k >= 0; k-- {
				if int(matrix[k][j]) == 48 {
					break
				}
				arr[j] += int(matrix[k][j]) - 48
			}
		}
		tmp := LargestRectangleArea(arr)
		if tmp > max {
			max = tmp
		}
	}
	return max
}

/*
86. 分隔链表
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.3 MB, 在所有 Go 提交中击败了65.14%的用户
通过测试用例：168 / 168
*/
func partition(head *ListNode, x int) *ListNode {
	small := &ListNode{}
	smallStart := small
	large := &ListNode{}
	largeStart := large

	for head != nil {
		if head.Val < x {
			small.Next = head
			small = small.Next
		} else {
			large.Next = head
			large = large.Next
		}
		head = head.Next
	}
	large.Next = nil
	small.Next = largeStart.Next
	return smallStart.Next
}

/*
87. 扰乱字符串
//抄的动态规划
给定两个字符串 T 和 S，如果 T 和 S 长度不一样，必定不能变来
如果长度一样，顶层字符串 S 能够划分S1 S2，同样字符串 T 也能够划分为T1 T2
则S1=>T1,S2=>T2或S1=>T2,S2=>T1
执行用时：8 ms, 在所有 Go 提交中击败了34.17%的用户
内存消耗：3.8 MB, 在所有 Go 提交中击败了65.83%的用户
通过测试用例：288 / 288
*/
func isScramble(s1 string, s2 string) bool {
	l := len(s1)
	dp := make([][][]bool, l)
	for i := range dp {
		dp[i] = make([][]bool, l)
		for j := range dp[i] {
			dp[i][j] = make([]bool, l+1)
		}
	}

	for i := 0; i < l; i++ {
		for j := 0; j < l; j++ {
			dp[i][j][1] = s1[i] == s2[j]
		}
	}

	// 枚举区间长度 2～l
	for len := 2; len <= l; len++ {
		// 枚举 S 中的起点位置
		for i := 0; i <= l-len; i++ {
			// 枚举 T 中的起点位置
			for j := 0; j <= l-len; j++ {
				// 枚举划分位置
				for k := 1; k <= len-1; k++ {
					// 第一种情况：S1 -> T1, S2 -> T2
					if dp[i][j][k] && dp[i+k][j+k][len-k] {
						dp[i][j][len] = true
						break
					}
					// 第二种情况：S1 -> T2, S2 -> T1
					// S1 起点 i，T2 起点 j + 前面那段长度 len-k ，S2 起点 i + 前面长度k
					if dp[i][j+len-k][k] && dp[i+k][j][len-k] {
						dp[i][j][len] = true
						break
					}
				}
			}
		}
	}
	return dp[0][0][l]
}

/*
88. 合并两个有序数组
正向双指针，额外[m]int{}
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.2 MB, 在所有 Go 提交中击败了25.00%的用户
通过测试用例：59 / 59
逆向双指针，不需要额外空间
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.2 MB, 在所有 Go 提交中击败了99.93%的用户
通过测试用例：59 / 59
*/
func Merge88(nums1 []int, m int, nums2 []int, n int) {
	/*tmp := append([]int{}, nums1[0:m]...)
	i, j := 0, 0
	for i < m || j < n {
		if i == m {
			nums1[i+j] = nums2[j]
			j++
		} else if j == n {
			nums1[i+j] = tmp[i]
			i++
		} else {
			if tmp[i] <= nums2[j] {
				nums1[i+j] = tmp[i]
				i++
			} else {
				nums1[i+j] = nums2[j]
				j++
			}
		}
	}*/
	//逆向双指针，不需要额外空间
	i := m - 1
	j := n - 1
	for i >= 0 || j >= 0 {
		if i < 0 {
			nums1[i+j+1] = nums2[j]
			j--
		} else if j < 0 {
			nums1[i+j+1] = nums1[i]
			i--
		} else {
			if nums1[i] > nums2[j] {
				nums1[i+j+1] = nums1[i]
				i--
			} else {
				nums1[i+j+1] = nums2[j]
				j--
			}
		}
	}
}

/*
89. 格雷编码
镜像反射法，n的格雷编码 = n-1的格雷编码 正序前加0 + 倒序前加1
执行用时：12 ms, 在所有 Go 提交中击败了21.89%的用户
内存消耗：6.8 MB, 在所有 Go 提交中击败了10.19%的用户
通过测试用例：16 / 16
*/
func GrayCode(n int) []int {
	if n == 0 {
		return []int{0}
	}
	if n == 1 {
		return []int{0, 1}
	}
	tmp := GrayCode(n - 1)
	res := append([]int{}, tmp...)
	for i := len(tmp) - 1; i >= 0; i-- {
		res = append(res, int(MyPow(2, n-1))+tmp[i])
	}
	return res
}

/*
90. 子集 II
参考40. 组合总和 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.3 MB, 在所有 Go 提交中击败了31.47%的用户
通过测试用例：20 / 20
*/
func SubsetsWithDup(nums []int) [][]int {
	sort.Ints(nums)
	//m记录candidates每个值出现的次数，0记录值，1记录次数
	//后续回溯m替代candidates
	var m [][2]int
	for _, v := range nums {
		if m == nil || v != m[len(m)-1][0] {
			m = append(m, [2]int{v, 1})
		} else {
			m[len(m)-1][1]++
		}
	}
	res := [][]int{}
	dfs90(m, 0, []int{}, &res)
	return res
}
func dfs90(m [][2]int, index int, tmp []int, res *[][]int) {
	if index == len(m) {
		*res = append(*res, append([]int{}, tmp...))
		return
	}
	n := m[index][1]

	//相同的处理，选1-n次
	for i := 1; i <= n; i++ {
		tmp = append(tmp, m[index][0])
		dfs90(m, index+1, tmp, res)
	}
	//tmp还原，减去前面append添加的最新n位
	tmp = tmp[:len(tmp)-n]
	dfs90(m, index+1, tmp, res)
}

/*
91. 解码方法
i+2项来源i项和i+1项
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：
1.9 MB, 在所有 Go 提交中击败了56.10%的用户
通过测试用例：269 / 269
*/
func numDecodings(s string) int {
	l := len(s)
	m := make([]int, l+1)
	m[0] = 1
	for i := 1; i <= l; i++ {
		if s[i-1] != '0' {
			m[i] += m[i-1]
		}
		if i > 1 && s[i-2] != '0' && ((s[i-2]-'0')*10+(s[i-1]-'0') <= 26) {
			m[i] += m[i-2]
		}
	}
	return m[l]
}

/*
92. 反转链表 II
头插法，在需要反转的区间里，每遍历到一个节点，让这个新节点来到反转部分的起始位置。
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了66.14%的用户
通过测试用例：44 / 44
*/
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	//left,right是第几个，不是.Val
	tmp := &ListNode{
		Next: head,
	}
	//记录left-1的node
	pre := tmp
	for i := 0; i < left-1; i++ {
		pre = pre.Next
	}
	//需要反转的
	index := pre.Next
	for i := 0; i < right-left; i++ {
		next := index.Next
		index.Next = next.Next
		next.Next = pre.Next
		pre.Next = next
	}

	return tmp.Next
}

/*
93. 复原 IP 地址
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了45.84%的用户
通过测试用例：145 / 145
*/
func RestoreIpAddresses(s string) []string {
	res := []string{}
	var dfs func(subRes []string, start int)
	dfs = func(subRes []string, start int) {
		if len(subRes) == 4 && start == len(s) {
			res = append(res, strings.Join(subRes, "."))
			return
		}
		if len(subRes) == 4 && start < len(s) {
			return
		}
		//选择1-3位
		for l := 1; l <= 3; l++ {
			if start+l > len(s) {
				return
			}
			if l > 1 && s[start] == '0' {
				return
			}
			str := s[start : start+l]
			if n, _ := strconv.Atoi(str); n > 255 {
				return
			}
			subRes = append(subRes, str)
			dfs(subRes, start+l)
			subRes = subRes[:len(subRes)-1]
		}
	}
	dfs([]string{}, 0)
	return res
}

/*
94. 二叉树的中序遍历
递归,迭代,morris算法
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了99.91%的用户
通过测试用例：70 / 70
*/
func InorderTraversal(root *TreeNode) []int {
	var res []int
	//递归
	/*var recursion func(n *TreeNode)
	recursion = func(n *TreeNode) {
		if n == nil {
			return
		}
		recursion(n.Left)
		res = append(res, n.Val)
		recursion(n.Right)
	}
	recursion(root)*/
	//迭代
	/*var iteration func(n *TreeNode)
	iteration = func(n *TreeNode) {
		var stack []*TreeNode
		for n != nil || len(stack) > 0 {
			for n != nil {
				stack = append(stack, n)
				n = n.Left
			}
			n = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res = append(res, n.Val)
			n = n.Right
		}
	}
	iteration(root)*/
	//Morris遍历算法
	for root != nil {
		if root.Left != nil {
			tmp := root.Left
			for tmp.Right != nil && tmp.Right != root {
				tmp = tmp.Right
			}
			if tmp.Right == nil {
				tmp.Right = root
				root = root.Left
			} else {
				//左子树完成遍历
				tmp.Right = nil
				res = append(res, root.Val)
				root = root.Right
			}
		} else {
			res = append(res, root.Val)
			root = root.Right
		}
	}
	return res
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

/*
95. 不同的二叉搜索树 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：4.2 MB, 在所有 Go 提交中击败了55.41%的用户
通过测试用例：8 / 8
*/
func generateTrees(n int) []*TreeNode {
	if n == 0 {
		return nil
	}
	return dfs95(1, n)
}
func dfs95(start, end int) []*TreeNode {
	if start > end {
		return []*TreeNode{nil}
	}
	var res []*TreeNode
	for i := start; i <= end; i++ {
		l := dfs95(start, i-1)
		r := dfs95(i+1, end)
		for j := 0; j < len(l); j++ {
			for k := 0; k < len(r); k++ {
				res = append(res, &TreeNode{
					Val:   i,
					Left:  l[j],
					Right: r[k],
				})
			}
		}
	}
	return res
}

/*
96. 不同的二叉搜索树
卡塔兰数,f(0) = 1,f(n+1) = f(n)*2*(2n+1)/n+2
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了79.57%的用户
通过测试用例：19 / 19
*/
func numTrees(n int) int {
	if n == 1 {
		return 1
	}
	return numTrees(n-1) * 2 * (2*(n-1) + 1) / ((n - 1) + 2)
}

/*
97. 交错字符串
f(i,j)表示s1前i个元素和s2前j个元素能否交错组合为s3前i+j个元素
f(i,j) = (f(i-1,j)&&s1[i-1]==s3[i+j-1]) || (f(i,j-1)&&s2[j-1]==s3[i+j-1])
滚动数组降低空间复杂度
非滚动数组
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了46.78%的用户
通过测试用例：106 / 106
滚动数组
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了96.14%的用户
通过测试用例：106 / 106
*/
func IsInterleave(s1 string, s2 string, s3 string) bool {
	l1, l2, l3 := len(s1), len(s2), len(s3)
	if l1+l2 != l3 {
		return false
	}

	//m := make([][]bool, l1+1)
	//for i := 0; i <= l1; i++ {
	//	m[i] = make([]bool, l2+1)
	//}
	//m[0][0] = true

	m := make([]bool, l2+1)
	m[0] = true

	for i := 0; i <= l1; i++ {
		for j := 0; j <= l2; j++ {
			//if i > 0 {
			//	m[i][j] = m[i][j] || (m[i-1][j] && s1[i-1] == s3[i+j-1])
			//}
			//if j > 0 {
			//	m[i][j] = m[i][j] || (m[i][j-1] && s2[j-1] == s3[i+j-1])
			//}

			if i > 0 {
				m[j] = m[j] && s1[i-1] == s3[i+j-1]
			}
			if j > 0 {
				m[j] = m[j] || m[j-1] && s2[j-1] == s3[i+j-1]
			}
			fmt.Println("i:", i)
			fmt.Println("j:", j)
			fmt.Println("m[j]:", m[j])
		}
	}
	//"aabcc", "dbbca", "aadbbcbcac"
	//return m[l1][l2]
	return m[l2]
}

/*
98. 验证二叉搜索树
递归
执行用时：4 ms, 在所有 Go 提交中击败了91.16%的用户
内存消耗：5.1 MB, 在所有 Go 提交中击败了16.23%的用户
通过测试用例：80 / 80
迭代
执行用时：4 ms, 在所有 Go 提交中击败了91.16%的用户
内存消耗：5 MB, 在所有 Go 提交中击败了32.60%的用户
通过测试用例：80 / 80
*/
func IsValidBST(root *TreeNode) bool {
	/*if root == nil {
		return true
	}
	arr := []int{}
	var recursion func(n *TreeNode) bool
	recursion = func(n *TreeNode) bool {
		if n == nil {
			return true
		}
		b := recursion(n.Left)
		if !b {
			return false
		}
		if len(arr) > 0 && arr[0] >= n.Val {
			return false
		}
		if len(arr) == 0 {
			arr = append(arr, n.Val)
		} else {
			arr[0] = n.Val
		}
		b = recursion(n.Right)
		return b
	}
	return recursion(root)*/

	if root == nil {
		return true
	}
	arr := []int{}
	var iteration func(n *TreeNode) bool
	iteration = func(n *TreeNode) bool {
		var stack []*TreeNode
		for n != nil || len(stack) > 0 {
			for n != nil {
				stack = append(stack, n)
				n = n.Left
			}
			n = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if len(arr) > 0 && arr[0] >= n.Val {
				return false
			}
			if len(arr) == 0 {
				arr = append(arr, n.Val)
			} else {
				arr[0] = n.Val
			}
			n = n.Right
		}
		return true
	}
	return iteration(root)
}

/*
99. 恢复二叉搜索树
Morris遍历算法，利用叶子节点空指针
执行用时：8 ms, 在所有 Go 提交中击败了83.65%的用户
内存消耗：5.5 MB, 在所有 Go 提交中击败了83.33%的用户
通过测试用例：1919 / 1919
*/
func recoverTree(root *TreeNode) {
	//x,y用于交换，pred用于比较
	var x, y, pred *TreeNode
	for root != nil {
		if root.Left != nil {
			tmp := root.Left
			for tmp.Right != nil && tmp.Right != root {
				tmp = tmp.Right
			}
			if tmp.Right == nil {
				tmp.Right = root
				root = root.Left
			} else {
				//左子树完成遍历
				if pred != nil && root.Val < pred.Val {
					y = root
					if x == nil {
						x = pred
					}
				}
				pred = root
				tmp.Right = nil
				root = root.Right
			}
		} else {
			if pred != nil && root.Val < pred.Val {
				y = root
				if x == nil {
					x = pred
				}
			}
			pred = root
			root = root.Right
		}
	}
	x.Val, y.Val = y.Val, x.Val
}

/*
100. 相同的树
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了14.31%的用户
通过测试用例：60 / 60
*/
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p != nil && q != nil {
		if p.Val != q.Val {
			return false
		}
		return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
	}
	return false
}
