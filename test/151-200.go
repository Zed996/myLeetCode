package test

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

/*
151. 颠倒字符串中的单词
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：6.6 MB, 在所有 Go 提交中击败了35.33%的用户
通过测试用例：58 / 58
*/
func ReverseWords(s string) string {
	var res string
	i := len(s) - 1
	start, end := -1, -1
	for ; i >= 0; i-- {
		if i > 0 && s[i] == ' ' && s[i-1] != ' ' {
			end = i
		}
		if i > 0 && s[i] != ' ' && s[i-1] == ' ' {
			start = i
		}
		if i == 0 && s[i] != ' ' {
			start = i
		}
		if i == len(s)-1 && s[i] != ' ' {
			end = i + 1
		}
		if start >= 0 && end >= 0 {
			res += s[start:end] + " "
			start, end = -1, -1
		}
	}
	return res[:len(res)-1]
}

/*
152. 乘积最大子数组
动态规划，以i结尾的数组最大乘积
fmax[i] = max{fmax[i-1]*nums[i],fmin[i-1]*nums[i],nums[i]}
fmin[i] = min{fmax[i-1]*nums[i],fmin[i-1]*nums[i],nums[i]}
执行用时：4 ms, 在所有 Go 提交中击败了93.83%的用户
内存消耗：3.2 MB, 在所有 Go 提交中击败了74.33%的用户
通过测试用例：188 / 188
*/
func MaxProduct(nums []int) int {
	res := nums[0]
	tmpMax, tmpMin := nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		tmp1 := max(max(tmpMin*nums[i], tmpMax*nums[i]), nums[i])
		tmp2 := min(min(tmpMin*nums[i], tmpMax*nums[i]), nums[i])
		tmpMax, tmpMin = tmp1, tmp2
		res = max(res, tmpMax)
	}
	return res
}

/*
153. 寻找旋转排序数组中的最小值
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.3 MB, 在所有 Go 提交中击败了57.19%的用户
通过测试用例：150 / 150
*/
func FindMin(nums []int) int {
	end := nums[len(nums)-1]
	l, r := 0, len(nums)-1
	mid := (l + r) / 2
	for l < r {
		fmt.Println("l:", l)
		fmt.Println("r:", r)
		mid = (l + r) / 2
		if nums[mid] > end {
			l = mid
		} else {
			r = mid
		}
		if l+1 == r {
			if nums[l] < nums[r] {
				return nums[l]
			} else {
				return nums[r]
			}
		}
	}
	return nums[mid]
}

/*
154. 寻找旋转排序数组中的最小值 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.9 MB, 在所有 Go 提交中击败了58.88%的用户
通过测试用例：193 / 193
*/
func findMin(nums []int) int {
	l, r := 0, len(nums)-1
	for l < r {
		mid := (l + r) / 2
		if nums[mid] > nums[r] {
			l = mid + 1
		} else if nums[mid] < nums[r] {
			r = mid
		} else {
			r--
		}
	}
	return nums[r]
}

/*
155. 最小栈
执行用时：16 ms, 在所有 Go 提交中击败了71.66%的用户
内存消耗：8 MB, 在所有 Go 提交中击败了90.23%的用户
通过测试用例：31 / 31
*/
type MinStack struct {
	Stack []int
	Min   int
}

func Constructor() MinStack {
	return MinStack{
		Stack: []int{},
		Min:   math.MinInt,
	}
}
func (this *MinStack) Push(val int) {
	this.Stack = append(this.Stack, val)
	if this.Min > val || len(this.Stack) == 1 {
		this.Min = val
	}
}
func (this *MinStack) Pop() {
	if this.Stack[len(this.Stack)-1] == this.Min {
		min := this.Stack[0]
		for i := 1; i < len(this.Stack)-1; i++ {
			if this.Stack[i] < min {
				min = this.Stack[i]
			}
		}
		this.Min = min
	}
	this.Stack = this.Stack[:len(this.Stack)-1]
}
func (this *MinStack) Top() int {
	return this.Stack[len(this.Stack)-1]
}
func (this *MinStack) GetMin() int {
	return this.Min
}

/*
156-159,161,163,170,186 -- 会员专享 todo
*/

/*
160. 相交链表
l1:a+c,l2:b+c;l1+l2=l2+l1
走到尽头见不到你，于是走过你来时的路，等到相遇时才发现，你也走过我来时的路。
执行用时：28 ms, 在所有 Go 提交中击败了76.52%的用户
内存消耗：7 MB, 在所有 Go 提交中击败了88.81%的用户
通过测试用例：39 / 39
*/
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	l1, l2 := headA, headB
	for l1 != l2 {
		if l1 == nil {
			l1 = headB
		} else {
			l1 = l1.Next
		}
		if l2 == nil {
			l2 = headA
		} else {
			l2 = l2.Next
		}
	}
	return l1
}

/*
162. 寻找峰值
始终选择大于边界一端进行二分，可以确保选择的区间一定存在峰值
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.6 MB, 在所有 Go 提交中击败了62.22%的用户
通过测试用例：63 / 63
*/
func findPeakElement(nums []int) int {
	l, r := 0, len(nums)-1
	for l < r {
		mid := (l + r) / 2
		if nums[mid] > nums[mid+1] {
			r = mid
		} else {
			l = mid + 1
		}
	}
	return r
}

/*
164. 最大间距
相邻数字的最大间距不会小于⌈(max−min)/(N−1)⌉
基于桶排序，桶数量d=⌊(max−min)/(N−1)⌋，维护每个桶内元素的最大值与最小值，max(相邻AB桶A最大与B最小差值)
执行用时：112 ms, 在所有 Go 提交中击败了84.67%的用户
内存消耗：10 MB, 在所有 Go 提交中击败了22.63%的用户
通过测试用例：41 / 41
*/
func MaximumGap(nums []int) int {
	if len(nums) < 2 {
		return 0
	}
	minN, maxN, l := nums[0], nums[0], len(nums)
	for i := 1; i < l; i++ {
		if nums[i] > maxN {
			maxN = nums[i]
			continue
		}
		if nums[i] < minN {
			minN = nums[i]
		}
	}
	d := (maxN - minN) / (l - 1)
	if d == 0 {
		d = 1
	}
	bucketSize := (maxN-minN)/d + 1
	type bucket struct {
		Min int
		Max int
	}
	buckets := make([]bucket, bucketSize)
	for i := range buckets {
		buckets[i] = bucket{-1, -1}
	}
	for i := 0; i < l; i++ {
		bid := (nums[i] - minN) / d
		if buckets[bid].Min == -1 {
			buckets[bid].Min = nums[i]
			buckets[bid].Max = nums[i]
		} else {
			buckets[bid].Min = min(buckets[bid].Min, nums[i])
			buckets[bid].Max = max(buckets[bid].Max, nums[i])
		}
	}

	var res int
	pre := 0
	for i := 1; i < len(buckets); i++ {
		if buckets[i].Min != -1 {
			res = max(buckets[i].Min-buckets[pre].Max, res)
			pre = i
		}
	}
	return res
}

/*
165. 比较版本号
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了51.89%的用户
通过测试用例：82 / 82
*/
func CompareVersion(version1 string, version2 string) int {
	arr1, arr2 := strings.Split(version1, "."), strings.Split(version2, ".")
	l1, l2 := len(arr1), len(arr2)
	l := l1
	if l1 < l2 {
		l = l2
	}
	for i := 0; i < l; i++ {
		if i < l1 && i < l2 {
			s1, _ := strconv.Atoi(arr1[i])
			s2, _ := strconv.Atoi(arr2[i])
			if s1 < s2 {
				return -1
			} else if s1 > s2 {
				return 1
			}
		}
		if i < l1 && i >= l2 {
			s1, _ := strconv.Atoi(arr1[i])
			if s1 > 0 {
				return 1
			}
		}
		if i >= l1 && i < l2 {
			s2, _ := strconv.Atoi(arr2[i])
			if s2 > 0 {
				return -1
			}
		}
	}
	return 0
}

/*
166. 分数到小数
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.7 MB, 在所有 Go 提交中击败了19.88%的用户
通过测试用例：39 / 39
*/
func FractionToDecimal(numerator int, denominator int) string {
	var res string
	negative := false
	if numerator*denominator < 0 {
		negative = true
	}
	numerator = abs(numerator)
	denominator = abs(denominator)
	integer := numerator / denominator
	remainder := numerator % denominator
	if remainder == 0 {
		res = strconv.Itoa(integer)
		if negative {
			return "-" + res
		}
		return res
	}
	decimal := remainder
	tmps := ""
	m := make(map[int]int, 10)
	for i := 0; i < 10; i++ {
		m[i] = -1
	}
	for i := 0; decimal != 0; i++ {
		decimal *= 10
		tmp := decimal / denominator
		//余数判断循环节
		if m[decimal] > 0 {
			l := i - m[decimal] + 1
			res = strconv.Itoa(integer) + "." + tmps[:len(tmps)-l] + "(" + tmps[len(tmps)-l:] + ")"
			if negative {
				return "-" + res
			}
			return res
		}
		m[decimal] = i + 1
		tmps += strconv.Itoa(tmp)
		decimal %= denominator
	}
	res = strconv.Itoa(integer) + "." + tmps
	if negative {
		return "-" + res
	}
	return res
}

/*
167. 两数之和 II - 输入有序数组
执行用时：8 ms, 在所有 Go 提交中击败了94.20%的用户
内存消耗：5.2 MB, 在所有 Go 提交中击败了68.41%的用户
通过测试用例：21 / 21
*/
func twoSum(numbers []int, target int) []int {
	i, j := 0, len(numbers)-1
	for {
		if numbers[i]+numbers[j] > target {
			j--
		} else if numbers[i]+numbers[j] < target {
			i++
		} else {
			return []int{i + 1, j + 1}
		}
	}
}

/*
168. Excel表列名称
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了39.06%的用户
通过测试用例：18 / 18
*/
func ConvertToTitle(columnNumber int) string {
	var res string
	/*for columnNumber > 0 {
		if columnNumber < 27 {
			res = string(rune(columnNumber+64)) + res
			return res
		}
		if columnNumber%26 == 0 {
			res = string(rune(90)) + res
			columnNumber -= 26
			columnNumber /= 26
		} else {
			res = string(rune((columnNumber%26)+64)) + res
			columnNumber /= 26
		}
	}*/
	//先-1
	for columnNumber > 0 {
		if columnNumber < 27 {
			res = string(rune(columnNumber+64)) + res
			return res
		}
		res = string(rune(((columnNumber-1)%26)+65)) + res
		columnNumber = (columnNumber - (columnNumber-1)%26) / 26
	}
	return res
}

/*
169. 多数元素
Boyer-Moore 投票算法
众数记为 +1，把其他数记为−1，将它们全部加起来，显然和大于 0
执行用时：12 ms, 在所有 Go 提交中击败了94.17%的用户
内存消耗：5.9 MB, 在所有 Go 提交中击败了76.20%的用户
通过测试用例：43 / 43
*/
func majorityElement(nums []int) int {
	candidate, count := 0, 0
	for i := 0; i < len(nums); i++ {
		if count == 0 {
			candidate = nums[i]
		}
		if candidate == nums[i] {
			count++
		} else {
			count--
		}
	}
	return candidate
}

/*
171. Excel 表列序号
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了64.74%的用户
通过测试用例：1002 / 1002
*/
func TitleToNumber(columnTitle string) int {
	res := 0
	for i := 0; i < len(columnTitle); i++ {
		tmp := int(columnTitle[i]) - 64
		res *= 26
		res += tmp
	}
	return res
}

/*
172. 阶乘后的零
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了7.39%的用户
通过测试用例：500 / 500
*/
func trailingZeroes(n int) int {
	res := 0
	for n > 0 {
		n /= 5
		res += n
	}
	return res
}

/*
173. 二叉搜索树迭代器
执行用时：16 ms, 在所有 Go 提交中击败了98.85%的用户
内存消耗：9.5 MB, 在所有 Go 提交中击败了83.91%的用户
通过测试用例：61 / 61
*/
type BSTIterator struct {
	stack []*TreeNode
	cur   *TreeNode
}

func Constructor173(root *TreeNode) BSTIterator {
	return BSTIterator{cur: root}
}
func (this *BSTIterator) Next() int {
	for node := this.cur; node != nil; node = node.Left {
		this.stack = append(this.stack, node)
	}
	this.cur, this.stack = this.stack[len(this.stack)-1], this.stack[:len(this.stack)-1]
	val := this.cur.Val
	this.cur = this.cur.Right
	return val
}
func (this *BSTIterator) HasNext() bool {
	return this.cur != nil || len(this.stack) > 0
}

/*
174. 地下城游戏
执行用时：4 ms, 在所有 Go 提交中击败了92.31%的用户
内存消耗：3.5 MB, 在所有 Go 提交中击败了45.25%的用户
通过测试用例：45 / 45
*/
func calculateMinimumHP(dungeon [][]int) int {
	r, c := len(dungeon), len(dungeon[0])
	f := make([][]int, r+1)
	for i := 0; i <= r; i++ {
		f[i] = make([]int, c+1)
		for j := 0; j <= c; j++ {
			f[i][j] = math.MaxInt32
		}
	}
	f[r][c-1], f[r-1][c] = 1, 1
	for i := r - 1; i >= 0; i-- {
		for j := c - 1; j >= 0; j-- {
			f[i][j] = max(min(f[i][j+1], f[i+1][j])-dungeon[i][j], 1)
		}
	}
	return f[0][0]
}

/*
175. 组合两个表
执行用时：477 ms, 在所有 MySQL 提交中击败了18.54%的用户
内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户
通过测试用例：8 / 8
*/
//select FirstName, LastName, City, State from Person left join Address on Person.PersonId = Address.PersonId;

/*
176. 第二高的薪水
执行用时：234 ms, 在所有 MySQL 提交中击败了72.24%的用户
内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户
通过测试用例：8 / 8
SELECT
    IFNULL(
      (SELECT DISTINCT Salary
       FROM Employee
       ORDER BY Salary DESC
        LIMIT 1 OFFSET 1),
    NULL) AS SecondHighestSalary
*/

/*
177. 第N高的薪水
执行用时：385 ms, 在所有 MySQL 提交中击败了44.79%的用户
内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户
通过测试用例：14 / 14
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
DECLARE m INT;
SET m = N - 1;
  RETURN (
      # Write your MySQL query statement below.
      SELECT ifnull( ( SELECT DISTINCT salary FROM Employee ORDER BY salary DESC LIMIT m, 1 ), NULL ) );
END
*/

/*
178. 分数排名
dense_rank() over ()
执行用时：281 ms, 在所有 MySQL 提交中击败了87.04%的用户
内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户
通过测试用例：10 / 10
select score, dense_rank() over (order by score desc) as 'rank'
from Scores
*/

/*
179. 最大数
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.2 MB, 在所有 Go 提交中击败了98.41%的用户
通过测试用例：230 / 230
*/
func largestNumber(nums []int) string {
	sort.Slice(nums, func(i, j int) bool {
		x, y := nums[i], nums[j]
		sx, sy := 10, 10
		for sx <= x {
			sx *= 10
		}
		for sy <= y {
			sy *= 10
		}
		return sy*x+y > sx*y+x
	})
	if nums[0] == 0 {
		return "0"
	}
	res := []byte{}
	for _, x := range nums {
		res = append(res, strconv.Itoa(x)...)
	}
	return string(res)
}

/*
180. 连续出现的数字
执行用时：584 ms, 在所有 MySQL 提交中击败了21.35%的用户
内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户
通过测试用例：21 / 21
SELECT DISTINCT
    l1.Num AS ConsecutiveNums
FROM
    Logs l1,
    Logs l2,
    Logs l3
WHERE
    l1.Id = l2.Id - 1
    AND l2.Id = l3.Id - 1
    AND l1.Num = l2.Num
    AND l2.Num = l3.Num
*/

/*
181-185 mysql todo
*/

/*
187. 重复的DNA序列
执行用时：16 ms, 在所有 Go 提交中击败了61.52%的用户
内存消耗：9.2 MB, 在所有 Go 提交中击败了37.50%的用户
通过测试用例：31 / 31
*/
func findRepeatedDnaSequences(s string) []string {
	var res []string
	m := make(map[string]int)
	for i := 0; i <= len(s)-10; i++ {
		m[s[i:i+10]]++
	}
	for k, v := range m {
		if v > 1 {
			res = append(res, k)
		}
	}
	return res
	//位运算更好 todo
	/*
		const L = 10
		var bin = map[byte]int{'A': 0, 'C': 1, 'G': 2, 'T': 3}

		func findRepeatedDnaSequences(s string) (ans []string) {
		    n := len(s)
		    if n <= L {
		        return
		    }
		    x := 0
		    for _, ch := range s[:L-1] {
		        x = x<<2 | bin[byte(ch)]
		    }
		    cnt := map[int]int{}
		    for i := 0; i <= n-L; i++ {
		        x = (x<<2 | bin[s[i+L-1]]) & (1<<(L*2) - 1)
		        cnt[x]++
		        if cnt[x] == 2 {
		            ans = append(ans, s[i:i+L])
		        }
		    }
		    return ans
		}
	*/
}

/*
188. 买卖股票的最佳时机 IV
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了77.89%的用户
通过测试用例：211 / 211
*/
func maxProfit(k int, prices []int) int {
	//buy[j]为前i天进行j笔交易，持有一只股票的最大利润
	//sell[j]为前i天进行j笔交易，不持有股票的最大利润
	//buy[j] = max(buy[j], sell[j]-prices[i])
	//sell[j] = max(sell[j], buy[j-1]+prices[i])
	// n天最多只能进行n/2笔交易
	l := len(prices)
	if l == 0 {
		return 0
	}
	k = min(k, l/2)
	buy := make([]int, k+1)
	sell := make([]int, k+1)
	for i := 1; i <= k; i++ {
		buy[i] = math.MinInt / 2
		sell[i] = math.MinInt / 2
	}
	buy[0] = -prices[0]

	for i := 1; i < l; i++ {
		buy[0] = max(buy[0], sell[0]-prices[i])
		for j := 1; j <= k; j++ {
			buy[j] = max(buy[j], sell[j]-prices[i])
			sell[j] = max(sell[j], buy[j-1]+prices[i])
		}
	}

	var res int
	for i := 0; i <= k; i++ {
		res = max(res, sell[i])
	}
	return res
}

/*
189. 轮转数组
翻转所有元素-翻转 [0,k mod n − 1] 区间的元素-翻转[k mod n,n−1] 区间的元素
执行用时：24 ms, 在所有 Go 提交中击败了82.84%的用户
内存消耗：7.8 MB, 在所有 Go 提交中击败了76.88%的用户
通过测试用例：38 / 38
*/
func rotate(nums []int, k int) {
	k = k % len(nums)
	if k == 0 {
		return
	}
	reserve(nums)
	reserve(nums[:k])
	reserve(nums[k:])
}
func reserve(arr []int) {
	for i, l := 0, len(arr); i < l/2; i++ {
		arr[i], arr[l-1-i] = arr[l-1-i], arr[i]
	}
}

/*
190. 颠倒二进制位 todo
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了97.77%的用户
通过测试用例：600 / 600
*/
func ReverseBits(num uint32) uint32 {
	var res uint32
	for i := 0; i < 32 && num > 0; i++ {
		res |= num & 1 << (31 - i)
		num >>= 1
	}
	return res
}

/*
191. 位1的个数
n & (n−1)，其运算结果恰为把 n 的二进制位中的最低位的 1 变为 0 之后的结果
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了20.90%的用户
通过测试用例：601 / 601
*/
func hammingWeight(num uint32) int {
	var res int
	for num != 0 {
		num &= num - 1
		res++
	}
	return res
}

/*
192. 统计词频
193. 有效电话号码
194. 转置文件
195. 第十行
//todo bash脚本
*/

/*
196. 删除重复的电子邮箱
执行用时：1038 ms, 在所有 MySQL 提交中击败了8.43%的用户
内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户
通过测试用例：22 / 22
*/
//DELETE p1 FROM Person p1,
//Person p2
//WHERE
//p1.Email = p2.Email AND p1.Id > p2.Id

/*
197. 上升的温度
执行用时：869 ms, 在所有 MySQL 提交中击败了5.01%的用户
内存消耗：0 B, 在所有 MySQL 提交中击败了100.00%的用户
通过测试用例：14 / 14
SELECT
    weather.id AS 'Id'
FROM
    weather
        JOIN
    weather w ON DATEDIFF(weather.recordDate, w.recordDate) = 1
        AND weather.Temperature > w.Temperature
;
*/

/*
198. 打家劫舍
f[i]存
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了8.15%的用户
通过测试用例：68 / 68
优化后xy存
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了72.44%的用户
通过测试用例：68 / 68
*/
func rob(nums []int) int {
	//f[i] = max(f[i-2]+nums[i],f[i-1])
	l := len(nums)
	if l == 0 {
		return 0
	}
	if l == 1 {
		return nums[0]
	}
	if l == 2 {
		return max(nums[1], nums[0])
	}
	/*f := make([]int, l)
	f[0] = nums[0]
	f[1] = max(nums[1], nums[0])
	for i := 2; i < l; i++ {
		f[i] = max(f[i-2]+nums[i], f[i-1])
	}
	return f[l-1]*/
	x, y := nums[0], max(nums[1], nums[0])
	for i := 2; i < l; i++ {
		tmp := y
		y = max(x+nums[i], y)
		x = tmp
	}
	return y
}

/*
199. 二叉树的右视图
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了93.11%的用户
通过测试用例：216 / 216
*/
func rightSideView(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var res []int
	stack := []*TreeNode{root}
	for len(stack) > 0 {
		res = append(res, stack[len(stack)-1].Val)
		var tmp []*TreeNode
		for i := 0; i < len(stack); i++ {
			if stack[i].Left != nil {
				tmp = append(tmp, stack[i].Left)
			}
			if stack[i].Right != nil {
				tmp = append(tmp, stack[i].Right)
			}
		}
		stack = tmp
	}
	return res
}

/*
200. 岛屿数量
执行用时：4 ms, 在所有 Go 提交中击败了83.78%的用户
内存消耗：3.7 MB, 在所有 Go 提交中击败了74.12%的用户
通过测试用例：49 / 49
*/
func numIslands(grid [][]byte) int {
	r, c := len(grid), len(grid[0])
	var dfs func(i, j int)
	dfs = func(i, j int) {
		if grid[i][j] == '1' {
			grid[i][j] = '2'
			if i > 0 {
				dfs(i-1, j)
			}
			if i < r-1 {
				dfs(i+1, j)
			}
			if j > 0 {
				dfs(i, j-1)
			}
			if j < c-1 {
				dfs(i, j+1)
			}
		}
	}
	var count int
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if grid[i][j] == '1' {
				count++
				dfs(i, j)
			}
		}
	}
	return count
}
