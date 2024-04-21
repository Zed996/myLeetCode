package test

import (
	"fmt"
	"math"
	"sort"
	"strconv"
)

/*
1. 两数之和
执行用时：4 ms, 在所有 Go 提交中击败了94.85%的用户
内存消耗：4.1 MB, 在所有 Go 提交中击败了20.64%的用户
通过测试用例：57 / 57
*/
func TwoSum(nums []int, target int) []int {
	m := make(map[int]int)
	for i, num := range nums {
		if v, ok := m[target-num]; ok {
			return []int{v, i}
		}
		m[num] = i
	}
	return nil
}

/*
2. 两数相加
执行用时：12 ms, 在所有 Go 提交中击败了43.73%的用户
内存消耗：4.4 MB, 在所有 Go 提交中击败了61.36%的用户
通过测试用例：1568 / 1568
*/
type ListNode struct {
	Val  int
	Next *ListNode
}

func AddTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	tmp := &ListNode{
		Val:  0,
		Next: nil,
	}
	res := tmp
	add := 0
	for l1 != nil || l2 != nil {
		if l1 != nil && l2 != nil {
			tmp.Val = (l1.Val + l2.Val + add) % 10
			add = (l1.Val + l2.Val + add) / 10
			l1 = l1.Next
			l2 = l2.Next
		} else if l2 != nil {
			tmp.Val = (l2.Val + add) % 10
			add = (l2.Val + add) / 10
			l2 = l2.Next
		} else if l1 != nil {
			tmp.Val = (l1.Val + add) % 10
			add = (l1.Val + add) / 10
			l1 = l1.Next
		}
		if l1 != nil || l2 != nil {
			tmp.Next = &ListNode{
				Val:  0,
				Next: nil,
			}
			tmp = tmp.Next
		}
	}
	if add > 0 {
		tmp.Next = &ListNode{
			Val:  add,
			Next: nil,
		}
	}
	return res
}

/*
3. 无重复字符的最长子串
执行用时：520 ms, 在所有 Go 提交中击败了5.71%的用户
内存消耗：6.7 MB, 在所有 Go 提交中击败了5.83%的用户
通过测试用例：987 / 987
*/
func LengthOfLongestSubstring(s string) int {
	if len(s) == 0 {
		return 0
	}
	//滑动窗口
	max := 1
	firstRepeat := 0
	for i := 0; i < len(s); i += firstRepeat {
		for j := i + max; j <= len(s); j++ {
			tmp := s[i:j]
			tmpMap := make(map[byte]int)
			repeat := false
			for k, v := range []byte(tmp) {
				if _, ok := tmpMap[v]; ok {
					firstRepeat = tmpMap[v]
					repeat = true
					break
				}
				tmpMap[v] = k
				if max < k+1 {
					max = k + 1
				}
			}
			if repeat {
				if firstRepeat == 0 {
					firstRepeat = 1
				}
				break
			}
		}
		if firstRepeat == 0 {
			firstRepeat = 1
		}
	}
	return max
}

/*
4. 寻找两个正序数组的中位数
 * 我们把数组 A 和数组 B 分别在 i 和 j 进行切割
 * 将 i 的左边和 j 的左边组合成「左半部分」，将 i 的右边和 j 的右边组合成「右半部分」
 * 当 A 数组和 B 数组的总长度是偶数时，如果我们能够保证
 * 左半部分的长度等于右半部分;左半部分最大的值小于等于右半部分最小的值 max ( A [ i - 1 ] , B [ j - 1 ]）） <= min ( A [ i ] , B [ j ]））
 *
 * 当 A 数组和 B 数组的总长度是奇数时，如果我们能够保证
 * 左半部分的长度比右半部分大1;左半部分最大的值小于等于右半部分最小的值 max ( A [ i - 1 ] , B [ j - 1 ]）） <= min ( A [ i ] , B [ j ]））
 *
 *
 * 执行用时：16 ms, 在所有 Go 提交中击败了38.06%的用户
 * 内存消耗：4.9 MB, 在所有 Go 提交中击败了79.08%的用户
 * 通过测试用例：2094 / 2094
*/
func FindMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	m := len(nums1)
	n := len(nums2)
	//保证m<=n
	if m > n {
		return FindMedianSortedArrays(nums2, nums1)
	}

	iMin := 0
	iMax := m
	for iMin <= iMax {
		i := (iMin + iMax) / 2
		j := (m+n+1)/2 - i
		if i != m && j != 0 && nums2[j-1] > nums1[i] {
			iMin = i + 1
		} else if i != 0 && j != n && nums2[j] < nums1[i-1] {
			iMax = i - 1
		} else {
			maxLeft := 0.0
			if i == 0 {
				maxLeft = float64(nums2[j-1])
			} else if j == 0 {
				maxLeft = float64(nums1[i-1])
			} else {
				maxLeft = math.Max(float64(nums1[i-1]), float64(nums2[j-1]))
			}
			// 奇数的话不需要考虑右半部分
			if (m+n)%2 == 1 {
				return maxLeft
			}

			minRight := 0.0
			if i == m {
				minRight = float64(nums2[j])
			} else if j == n {
				minRight = float64(nums1[i])
			} else {
				minRight = math.Min(float64(nums1[i]), float64(nums2[j]))
			}
			return (maxLeft + minRight) / 2.0
		}
	}
	return 0.0
}

/*
5. 最长回文子串
Manacher算法
执行用时：8 ms, 在所有 Go 提交中击败了68.72%的用户
内存消耗：6.8 MB, 在所有 Go 提交中击败了42.57%的用户
通过测试用例：180 / 180
*/
func LongestPalindrome(s string) string {
	if len(s) == 0 {
		return ""
	}
	bArr := []byte(s)
	sArr := []string{"^"}
	p := []int{0, 0, 0}
	for i := 0; i < len(s); i++ {
		sArr = append(sArr, "#")
		sArr = append(sArr, string(bArr[i]))
		p = append(p, 0)
		p = append(p, 0)
	}
	sArr = append(sArr, "#")
	sArr = append(sArr, "$")

	//c对称中心key
	//r回文串最右key
	c := 0
	r := 0
	for i := 1; i < len(sArr)-1; i++ {
		j := 2*c - i
		if r > i {
			p[i] = int(math.Min(float64(r-i), float64(p[j])))
		} else {
			p[i] = 0
		}

		//中心扩展
		for sArr[i+1+p[i]] == sArr[i-1-p[i]] {
			p[i]++
		}

		// 判断是否需要更新r
		if i+p[i] > r {
			c = i
			r = i + p[i]
		}
	}

	// 找出p的最大值
	maxLen := 0
	centerIndex := 0
	for i := 1; i < len(sArr)-1; i++ {
		if p[i] > maxLen {
			maxLen = p[i]
			centerIndex = i
		}
	}

	//最开始讲的求原字符串下标
	start := (centerIndex - maxLen) / 2
	return s[start : start+maxLen]
}

/*
6. Z 字形变换
二维矩阵
执行用时：60 ms, 在所有 Go 提交中击败了6.03%的用户
内存消耗：15.9 MB, 在所有 Go 提交中击败了4.99%的用户
通过测试用例：1157 / 1157

直接构造
执行用时：4 ms, 在所有 Go 提交中击败了91.81%的用户
内存消耗：3.7 MB, 在所有 Go 提交中击败了94.88%的用户
通过测试用例：1157 / 1157
*/
func Convert(s string, numRows int) string {
	/*
		    二维矩阵
		    if len(s) <= numRows || numRows == 1 {
				return s
			}
			arr := make([][]string, numRows)
			l := len(s) / (2*numRows - 2)
			remainder := len(s) % (2*numRows - 2)
			for i := 0; i < numRows; i++ {
				if remainder/numRows > 0 {
					arr[i] = make([]string, l*(numRows-1)+numRows-i)
				} else {
		            arr[i] = make([]string, l*(numRows-1)+1)
				}
			}
			r := 0
			c := 0
			down := true
			for i := 0; i < len(s); i++ {
				arr[r][c] = s[i : i+1]
				if r == numRows-1 {
					down = false
				}
				if r == 0 {
					down = true
				}
				if down {
					r++
				} else {
					r--
					c++
				}
			}

			var res string
			for _, v := range arr {
				for _, v1 := range v {
					if v1 != "" {
						res += v1
					}
				}
			}
			return res
	*/

	//直接构造
	if len(s) <= numRows || numRows == 1 {
		return s
	}
	//2*numRows-2为周期长度
	t := 2*numRows - 2
	res := make([]byte, 0, len(s))
	for i := 0; i < numRows; i++ { // 枚举矩阵的行
		for j := 0; j+i < len(s); j += t { // 枚举每个周期的起始下标
			res = append(res, s[j+i]) // 当前周期的第一个字符
			if i > 0 && i < numRows-1 && j+t-i < len(s) {
				res = append(res, s[j+t-i]) // 当前周期的第二个字符
			}
		}
	}
	return string(res)
}

/*
7. 整数反转
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了99.96%的用户
通过测试用例：1032 / 1032
*/
func Reverse(x int) int {
	res := 0
	//记录上次的，判断溢出
	last := 0
	for x != 0 {
		tmp := x % 10
		last = res
		res = res*10 + tmp
		if last != res/10 || res > 2147483647 || res < -(2147483648) {
			return 0
		}
		x /= 10
	}
	return res
}

/*
8. 字符串转换整数 (atoi)
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了13.43%的用户
通过测试用例：1082 / 1082
*/
func MyAtoi(s string) int {
	if s == "" {
		return 0
	}
	//ascII: + 43;- 45;0-9 48-57; (空格) 32;
	r := []rune(s)

	start := -1
	var resArr []rune
	for k, v := range r {
		if int(v) != 32 {
			if start == -1 {
				if !(int(v) >= 48 && int(v) <= 57) && int(v) != 43 && int(v) != 45 {
					return 0
				}
				if int(v) == 43 || int(v) == 45 {
					if len(r) == k+1 {
						return 0
					}
					if !(int(r[k+1]) >= 48 && int(r[k+1]) <= 57) {
						return 0
					}
				}
			}
			if int(v) >= 48 && int(v) <= 57 {
				if start == -1 {
					start = k
				}
				resArr = append(resArr, v)
				if k == len(r)-1 || int(r[k+1]) < 48 || int(r[k+1]) > 57 {
					break
				}
			}
		}
	}
	res := 0
	last := res
	for i := 0; i < len(resArr); i++ {
		res = res*10 + (int(resArr[i]) - 48)
		if res/10 != last {
			if start > 0 && int(r[start-1]) == 45 {
				return -2147483648
			} else {
				return 2147483647
			}
		}
		last = res
	}
	if start > 0 && int(r[start-1]) == 45 {
		res = -1 * res
	}
	if res > 2147483647 {
		res = 2147483647
	}
	if res < -(2147483648) {
		res = -(2147483648)
	}
	return res
}

/*
9. 回文数
执行用时：20 ms, 在所有 Go 提交中击败了31.13%的用户
内存消耗：4.4 MB, 在所有 Go 提交中击败了73.02%的用户
通过测试用例： 11510 / 11510
*/
func IsPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	if x == 0 {
		return true
	}
	origin := x
	trans := 0
	for x != 0 {
		trans = trans*10 + x%10
		x = x / 10
	}
	if trans == origin {
		return true
	}
	return false
}

/*
10. 正则表达式匹配
动态规划
s[i] == p[j]: dp[i][j] = dp[i-1][j-1]
p[j] == '*' & s[i] != p[j−1]: dp[i][j] = dp[i][j-2]
p[j] == '*' & s[i] == p[j−1]: dp[i][j] = dp[i][j-2] || dp[i-1][j]
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了83.52%的用户
通过测试用例：353 / 353
*/
func IsMatch10(s string, p string) bool {
	m, n := len(s), len(p)
	matches := func(i, j int) bool {
		if i == 0 {
			return false
		}
		if p[j-1] == '.' {
			return true
		}
		return s[i-1] == p[j-1]
	}

	f := make([][]bool, m+1)
	for i := 0; i < len(f); i++ {
		f[i] = make([]bool, n+1)
	}
	f[0][0] = true
	for i := 0; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == '*' {
				f[i][j] = f[i][j] || f[i][j-2]
				if matches(i, j-1) {
					f[i][j] = f[i][j] || f[i-1][j]
				}
			} else if matches(i, j) {
				f[i][j] = f[i][j] || f[i-1][j-1]
			}
		}
	}
	return f[m][n]
}

/*
11. 盛最多水的容器
执行用时：76 ms, 在所有 Go 提交中击败了19.75%的用户
内存消耗：8.5 MB, 在所有 Go 提交中击败了37.75%的用户
通过测试用例：60 / 60
*/
func MaxArea(height []int) int {
	//双指针
	max := 0
	i := 0
	j := len(height) - 1
	for i < j {
		max = int(math.Max(math.Min(float64(height[i]), float64(height[j]))*float64(j-i), float64(max)))
		if height[i] > height[j] {
			j--
		} else {
			i++
		}
	}
	return max
}

/*
12. 整数转罗马数字
执行用时：12 ms, 在所有 Go 提交中击败了31.78%的用户
内存消耗：3 MB, 在所有 Go 提交中击败了98.33%的用户
通过测试用例：3999 / 3999
*/
func IntToRoman(num int) string {
	//硬编码表
	thousands := []string{"", "M", "MM", "MMM"}
	hundreds := []string{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"}
	tens := []string{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"}
	ones := []string{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}
	return thousands[num/1000] + hundreds[num%1000/100] + tens[num%100/10] + ones[num%10]
}

/*
13. 罗马数字转整数
根据12
执行用时：260 ms, 在所有 Go 提交中击败了6.78%的用户
内存消耗：2.8 MB, 在所有 Go 提交中击败了15.66%的用户
通过测试用例：3999 / 3999

普通写法
执行用时：4 ms, 在所有 Go 提交中击败了88.82%的用户
内存消耗：2.7 MB, 在所有 Go 提交中击败了99.79%的用户
通过测试用例：3999 / 3999
*/
func RomanToInt(s string) int {
	/*
		//辣鸡代码，根据上一题写的
		thousands := []string{"", "M", "MM", "MMM"}
		hundreds := []string{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"}
		tens := []string{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"}
		ones := []string{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}
		for num := 1; num < 4000; num++ {
			if s == (thousands[num/1000] + hundreds[num%1000/100] + tens[num%100/10] + ones[num%10]) {
				return num
			}
		}
		return 0
	*/
	//把一个小值放在大值的左边，就是做减法，否则为加法。最后一位必定加
	stringToInt := make(map[string]int)
	stringToInt["I"] = 1
	stringToInt["V"] = 5
	stringToInt["X"] = 10
	stringToInt["L"] = 50
	stringToInt["C"] = 100
	stringToInt["D"] = 500
	stringToInt["M"] = 1000

	//直接初始化最后一位的值
	res := stringToInt[string(s[len(s)-1])]

	for i := 1; i < len(s); i++ {
		if stringToInt[string(s[i])] > stringToInt[string(s[i-1])] {
			res -= stringToInt[string(s[i-1])]
		} else {
			res += stringToInt[string(s[i-1])]
		}
	}

	return res
}

/*
14. 最长公共前缀
执行用时：4 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：5.4 MB, 在所有 Go 提交中击败了5.06%的用户
通过测试用例：124 / 124
*/
func LongestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}

	longest := func(a, b string) string {
		if len(a) == 0 || len(b) == 0 {
			return ""
		}
		if len(a) > len(b) {
			tmp := b
			b = a
			a = tmp
		}
		res := ""
		for i := 0; i < len(a); i++ {
			if a[i] != b[i] {
				break
			} else {
				res += string(a[i])
			}
		}
		return res
	}

	res := strs[0]
	for i := 1; i < len(strs); i++ {
		res = longest(res, strs[i])
		if res == "" {
			return ""
		}
	}
	return res
}

/*
15. 三数之和
执行用时：28 ms, 在所有 Go 提交中击败了89.82%的用户
内存消耗：7.3 MB, 在所有 Go 提交中击败了57.72%的用户
通过测试用例：318 / 318
*/
func ThreeSum(nums []int) [][]int {
	if len(nums) < 3 {
		return [][]int{}
	}
	sort.Ints(nums)
	res := [][]int{}
	/*若
	nums[i]>0nums[i]>0：因为已经排序好，所以后面不可能有三个数加和等于0，直接返回结果。
	对于重复元素：跳过，避免出现重复解
	令左指针 L=i+1L=i+1，右指针 R=n-1R=n−1，当 L<RL<R 时，执行循环：
		当 nums[i]+nums[L]+nums[R]==0nums[i]+nums[L]+nums[R]==0，执行循环，判断左界和右界是否和下一位置重复，去除重复解。并同时将 L,RL,R 移到下一位置，寻找新的解
		若和大于0，说明 nums[R]nums[R] 太大，RR 左移
		若和小于0，说明 nums[L]nums[L] 太小，LL 右移
	*/

	for i := 0; i < len(nums)-2; i++ {
		if nums[i] > 0 {
			return res
		}
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		j := i + 1
		k := len(nums) - 1
		for j < k {
			if nums[i]+nums[j]+nums[k] == 0 {
				res = append(res, []int{nums[i], nums[j], nums[k]})
				for j+1 < k && nums[j] == nums[j+1] {
					j++
				}
				for k-1 > j && nums[k] == nums[k-1] {
					k--
				}
				j++
				k--
			} else if nums[i]+nums[j]+nums[k] > 0 {
				k--
			} else {
				j++
			}
		}
	}
	return res
}

/*
16. 最接近的三数之和
执行用时：136 ms, 在所有 Go 提交中击败了10.86%的用户
内存消耗：6.4 MB, 在所有 Go 提交中击败了10.81%的用户
通过测试用例：381 / 381
*/
func ThreeSumClosest(nums []int, target int) int {
	sort.Ints(nums)
	res := nums[0] + nums[1] + nums[2]
	if res == target {
		return target
	}

	for i := 0; i < len(nums)-2; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		j := i + 1
		k := len(nums) - 1
		for j < k {
			if nums[i]+nums[j]+nums[k] == target {
				return target
			} else if nums[i]+nums[j]+nums[k] > target {
				if int(math.Abs(float64(res-target))) > (nums[i] + nums[j] + nums[k] - target) {
					res = nums[i] + nums[j] + nums[k]
				}
				for j+1 < k && nums[j] == nums[j+1] && k-1 > j && nums[k] == nums[k-1] {
					j++
					k--
				}
				k--
			} else {
				if int(math.Abs(float64(res-target))) > (target - nums[i] - nums[j] - nums[k]) {
					res = nums[i] + nums[j] + nums[k]
				}
				for j+1 < k && nums[j] == nums[j+1] && k-1 > j && nums[k] == nums[k-1] {
					j++
					k--
				}
				j++
			}
		}
	}

	return res
}

/*
17. 电话号码的字母组合
回溯法，每步向下推进至路径终点
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了36.43%的用户
通过测试用例：25 / 25
*/
func LetterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	combinations = []string{}
	backtrack17(digits, 0, "")
	return combinations
}

var phoneMap = map[string]string{
	"2": "abc",
	"3": "def",
	"4": "ghi",
	"5": "jkl",
	"6": "mno",
	"7": "pqrs",
	"8": "tuv",
	"9": "wxyz",
}
var combinations []string

func backtrack17(digits string, index int, combination string) {
	if index == len(digits) {
		combinations = append(combinations, combination)
	} else {
		for i := 0; i < len(phoneMap[string(digits[index])]); i++ {
			backtrack17(digits, index+1, combination+string(phoneMap[string(digits[index])][i]))
		}
	}
}

/*
18. 四数之和
后两个双指针，前两个两层for遍历
执行用时：12 ms, 在所有 Go 提交中击败了48.15%的用户
内存消耗：2.6 MB, 在所有 Go 提交中击败了70.41%的用户
通过测试用例：291 / 291
*/
func FourSum(nums []int, target int) [][]int {
	sort.Ints(nums)
	//fmt.Println(nums)
	res := [][]int{}
	for i := 0; i < len(nums)-3; i++ {
		if nums[i] >= ((target / 4) + 1) {
			return res
		}
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		for j := i + 1; j < len(nums)-2; j++ {
			if j > i+1 && nums[j] == nums[j-1] {
				continue
			}
			l := j + 1
			r := len(nums) - 1
			for l < r {
				//fmt.Println(i, j, l, r)
				if nums[i]+nums[j]+nums[l]+nums[r] == target {
					res = append(res, []int{nums[i], nums[j], nums[l], nums[r]})
					for l+1 < r && nums[l] == nums[l+1] {
						l++
					}
					for r-1 > j && nums[r] == nums[r-1] {
						r--
					}
					l++
					r--
				} else if nums[i]+nums[j]+nums[r]+nums[l] > target {
					r--
				} else {
					l++
				}
			}
		}
	}
	return res
}

/*
19. 删除链表的倒数第 N 个结点
双指针,开始0和n，移动至倒数n+1和倒数1
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了99.94%的用户
通过测试用例：208 / 208
*/
func RemoveNthFromEnd(head *ListNode, n int) *ListNode {
	//添加一个0节点
	tmp := &ListNode{
		Val:  0,
		Next: head,
	}
	first := head
	second := tmp
	for i := 0; i < n; i++ {
		first = first.Next
	}
	for first != nil {
		first = first.Next
		second = second.Next
	}
	second.Next = second.Next.Next
	return tmp.Next
}

/*
20. 有效的括号
模拟栈实现
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：6.4 MB, 在所有 Go 提交中击败了5.09%的用户
通过测试用例：91 / 91
*/
func IsValid(s string) bool {
	if len(s)%2 != 0 {
		return false
	}
	stack := ""
	for i := 0; i < len(s); i++ {
		switch string(s[i]) {
		case "]":
			if len(stack) > 0 && stack[len(stack)-1:] == "[" {
				stack = stack[:len(stack)-1]
			} else {
				stack += "]"
			}
		case "}":
			if len(stack) > 0 && stack[len(stack)-1:] == "{" {
				stack = stack[:len(stack)-1]
			} else {
				stack += "}"
			}
		case ")":
			if len(stack) > 0 && stack[len(stack)-1:] == "(" {
				stack = stack[:len(stack)-1]
			} else {
				stack += ")"
			}
		default:
			stack += string(s[i])
		}
	}
	return len(stack) == 0
}

/*
21. 合并两个有序链表
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了99.93%的用户
通过测试用例：208 / 208
*/
func MergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}
	if list2 == nil {
		return list1
	}

	//认为1首位更小
	if list1.Val > list2.Val {
		return MergeTwoLists(list2, list1)
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

/*
22. 括号生成
dfs或回溯；回溯有顺序问题无法通过
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.6 MB, 在所有 Go 提交中击败了71.08%的用户
通过测试用例：8 / 8
*/
func GenerateParenthesis(n int) []string {
	//if n == 1 {
	//	return []string{"()"}
	//}
	//return backtrack22(1, n, []string{"()"})
	res := []string{}
	dfs22(0, 0, n, "", &res)
	return res
}
func dfs22(left int, right int, n int, str string, res *[]string) {
	if left == n && right == n {
		*res = append(*res, str)
		return
	}
	if left < n {
		dfs22(left+1, right, n, str+"(", res)
	}
	if right < n && left > right {
		dfs22(left, right+1, n, str+")", res)
	}
}

/*
func backtrack22(index int, n int, sArr []string) []string {
	if index == n {
		return sArr
	}
	var tmp []string
	for _, v := range sArr {
		tmp = append(tmp, "("+v+")")
		if "()"+v != v+"()" {
			tmp = append(tmp, v+"()")
			tmp = append(tmp, "()"+v)
		} else {
			tmp = append(tmp, "()"+v)
		}
	}
	return backtrack22(index+1, n, tmp)
}*/

/*
23. 合并K个升序链表
根据21，分治
执行用时：8 ms, 在所有 Go 提交中击败了87.45%的用户
内存消耗：5.1 MB, 在所有 Go 提交中击败了86.89%的用户
通过测试用例：133 / 133
*/
func MergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 1 {
		return lists[0]
	}
	if len(lists) == 0 {
		return nil
	}
	return MergeTwoLists(MergeKLists(lists[:len(lists)/2]), MergeKLists(lists[len(lists)/2:]))
}

/*
24. 两两交换链表中的节点
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了58.50%的用户
通过测试用例：55 / 55
*/
func SwapPairs(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	tmp := &ListNode{
		Val:  0,
		Next: head,
	}
	first := tmp
	for first.Next.Next != nil {
		//fmt.Println("first.Val:", first.Val)
		n1 := first.Next
		n2 := first.Next.Next
		n3 := first.Next.Next.Next
		if n3 == nil {
			first.Next = n2
			n2.Next = n1
			n1.Next = n3
			return tmp.Next
		}
		first.Next = n2
		n2.Next = n1
		n1.Next = n3
		first = n2.Next
		//fmt.Println("trans:")
		//tmp1 := tmp
		//for tmp1 != nil {
		//	fmt.Println("tmp.Val:", tmp1.Val)
		//	tmp1 = tmp1.Next
		//}
		//fmt.Println("first.Val:", first.Val)
		//fmt.Println()
	}
	return tmp.Next
}

/*
25. K 个一组翻转链表
执行用时：4 ms, 在所有 Go 提交中击败了87.03%的用户
内存消耗：3.4 MB, 在所有 Go 提交中击败了76.27%的用户
通过测试用例：62 / 62
*/
func ReverseKGroup(head *ListNode, k int) *ListNode {
	tmp := &ListNode{
		Next: head,
	}
	if k == 1 {
		return tmp.Next
	}
	pre := tmp

	for head != nil {
		tail := pre
		for i := 0; i < k; i++ {
			tail = tail.Next
			if tail == nil {
				return tmp.Next
			}
		}
		next := tail.Next
		head, tail = ReverseHeadTail(head, tail)
		pre.Next = head
		tail.Next = next
		pre = tail
		head = tail.Next
	}
	return tmp.Next
}

func ReverseHeadTail(head, tail *ListNode) (*ListNode, *ListNode) {
	n1 := head
	n2 := head.Next
	for n2 != nil {
		if n2 == tail {
			n2.Next = n1
			break
		} else {
			tmp := n2.Next
			n2.Next = n1
			n1 = n2
			n2 = tmp
		}
	}
	return tail, head
}

/*
26. 删除有序数组中的重复项
类似双指针
执行用时：12 ms, 在所有 Go 提交中击败了14.59%的用户
内存消耗：4.2 MB, 在所有 Go 提交中击败了63.40%的用户
通过测试用例：361 / 361
*/
func RemoveDuplicates(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	l := 1
	for i := 1; i < len(nums); i++ {
		if nums[i-1] != nums[i] {
			nums[l] = nums[i]
			l++
		}
	}
	return l + 1
}

/*
27. 移除元素
同上，双指针
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2 MB, 在所有 Go 提交中击败了99.97%的用户
通过测试用例：113 / 113
*/
func RemoveElement(nums []int, val int) int {
	if len(nums) == 0 {
		return 0
	}

	l := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != val {
			nums[l] = nums[i]
			l++
		}
	}
	return l
}

/*
28. 实现 strStr()
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了74.78%的用户
通过测试用例：75 / 75
*/
func StrStr(haystack string, needle string) int {
	if len(needle) == 0 {
		return 0
	}
	if len(haystack) < len(needle) {
		return -1
	}
	if len(haystack) == len(needle) && haystack == needle {
		return 0
	}

	for i := 0; i <= len(haystack)-len(needle); i++ {
		if needle == haystack[i:len(needle)+i] {
			return i
		}
	}
	return -1
}

/*
29. 两数相除
使用快速相乘法：例如除数是x，那么计算过程如下：x*1 -> x*2 -> x*4 -> x*8
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.3 MB, 在所有 Go 提交中击败了5.67%的用户
通过测试用例：992 / 992
*/
func Divide(dividend int, divisor int) int {
	if dividend == -2147483648 && divisor == -1 {
		return 2147483647
	}
	if math.Abs(float64(dividend)) < math.Abs(float64(divisor)) {
		return 0
	}
	//记录符号位，转为负数计算，防溢出
	sign := 1
	if (dividend < 0 && divisor > 0) || (dividend > 0 && divisor < 0) {
		sign = -1
	}
	dividend = int(0 - math.Abs(float64(dividend)))
	divisor = int(0 - math.Abs(float64(divisor)))

	tmp := divisor
	i := 0
	resMap := make(map[int]int)
	power := make(map[int]int)
	for {
		resMap[i] = tmp
		tmp = tmp + tmp
		if tmp < dividend {
			break
		}
		i++
	}
	for ; i >= 0; i-- {
		dividend -= resMap[i]
		power[i] = 1
		for i > 0 && dividend > resMap[i-1] {
			i--
			power[i] = 0
		}
	}
	res := 0
	p := 1
	for j := 0; j < len(power); j++ {
		if power[j] > 0 {
			res += p
			p = p + p
		} else {
			p = p + p
		}
	}
	if sign == 1 {
		return res
	} else {
		return 0 - res
	}
}

/*
30. 串联所有单词的子串
//辣鸡代码，一个一个判断
执行用时：176 ms, 在所有 Go 提交中击败了16.72%的用户
内存消耗：7.4 MB, 在所有 Go 提交中击败了18.01%的用户
通过测试用例：177 / 177
*/
func FindSubstring(s string, words []string) []int {
	ls := len(s)
	lw := len(words[0])
	lwarr := len(words)
	if ls < lw*lwarr {
		return []int{}
	}
	var res []int
	for i := 0; i <= ls-lw*lwarr; i++ {
		//fmt.Println("i:", i)
		tmpMap := make(map[string]int)
		for _, v := range words {
			tmpMap[v]++
		}
		for j := 0; j < lwarr; j++ {
			//fmt.Println("tmpMap[s[i+j*lw:i+(j+1)*lw]]:", tmpMap[s[i+j*lw:i+(j+1)*lw]])
			//fmt.Println("s[i+j*lw:i+(j+1)*lw]:", s[i+j*lw:i+(j+1)*lw])
			_, ok := tmpMap[s[i+j*lw:i+(j+1)*lw]]
			if !ok {
				break
			} else {
				tmpMap[s[i+j*lw:i+(j+1)*lw]]--
			}
		}
		zero := true
		for _, v := range tmpMap {
			if v != 0 {
				zero = false
			}
		}
		if zero {
			res = append(res, i)
		}
	}
	return res
}

/*
31. 下一个排列
1. 从后向前查找第一个相邻升序的元素对 (i-1,i)，满足 A[i-1] < A[i]。此时 [i,end) 必然是降序
2. 在 [i,end) 从后向前查找第一个满足 A[i-1] < A[k] 的 k。
3. 将 A[i-1] 与 A[k] 交换
4. 这时 [i,end) 必然是降序，逆置 [i,end)，使其升序
5. 如果无交换，说明逆序数组，转为正序
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.3 MB, 在所有 Go 提交中击败了63.90%的用户
通过测试用例：265 / 265
*/
func NextPermutation(nums []int) {
	swap := false
	for i := len(nums) - 1; i > 0; i-- {
		if nums[i] > nums[i-1] {
			k := len(nums) - 1
			for ; k >= i; k-- {
				if nums[i-1] < nums[k] {
					break
				}
			}
			tmp := nums[i-1]
			nums[i-1] = nums[k]
			nums[k] = tmp
			swap = true
			sort.Ints(nums[i:])
			break
		}
	}
	if !swap {
		sort.Ints(nums)
	}
	return
}

/*
32. 最长有效括号
遍历，记录左右数量，相等返回长度，不符合要求重置左右数量
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了100.00%的用户
通过测试用例：231 / 231
*/
func LongestValidParentheses(s string) int {
	max := 0
	left := 0
	right := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			left++
		} else {
			right++
		}
		if left == right && left*2 > max {
			max = left * 2
		} else if right > left {
			left = 0
			right = 0
		}
	}
	//倒叙，处理左括号的数量始终大于右括号的数量情况，(()
	left = 0
	right = 0
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '(' {
			left++
		} else {
			right++
		}
		if left == right && left*2 > max {
			max = left * 2
		}
		if left > right {
			left = 0
			right = 0
		}
	}
	return max
}

/*
33. 搜索旋转排序数组
先找实际最小的key记作zero，两个有序数组，选一个二分查找target
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了57.96%的用户
通过测试用例：195 / 195
*/
func Search(nums []int, target int) int {
	if len(nums) == 1 && nums[0] == target {
		return 0
	}
	if len(nums) == 2 {
		if nums[0] == target {
			return 0
		} else if nums[1] == target {
			return 1
		} else {
			return -1
		}
	}
	left := 0
	right := len(nums) - 1
	zero := 0
	if nums[left] > nums[right] {
		for left < right {
			//fmt.Println("left:", left)
			//fmt.Println("right:", right)
			if nums[(left+right)/2-1] > nums[(left+right)/2] {
				zero = (left + right) / 2
				break
			}
			if nums[left] < nums[(left+right)/2] {
				left = (left + right) / 2
			} else {
				right = (left + right) / 2
			}
			if left+1 == right {
				if nums[left] > nums[right] {
					zero = right
					break
				} else {
					zero = left
					break
				}
			}
		}
	}

	//fmt.Println("zero:", zero)

	switch zero {
	case 0:
		left = 0
		right = len(nums) - 1
	case 1:
		if target == nums[0] {
			return 0
		}
		left = 1
		right = len(nums) - 1
	case len(nums) - 1:
		if target == nums[len(nums)-1] {
			return len(nums) - 1
		}
		left = 0
		right = len(nums) - 2
	default:
		if target > nums[0] {
			//左侧有序
			left = 0
			right = zero - 1
		} else if target < nums[0] {
			//右侧有序
			left = zero
			right = len(nums) - 1
		} else {
			return 0
		}
	}
	//fmt.Println("left:", left)
	//fmt.Println("right:", right)

	for left < right {
		if nums[(left+right)/2] == target {
			return (left + right) / 2
		}
		if nums[(left+right)/2] > target {
			right = (left + right) / 2
		} else {
			left = (left + right) / 2
		}
		if left+1 == right {
			if nums[left] == target {
				return left
			} else if nums[right] == target {
				return right
			} else {
				break
			}
		}
	}

	return -1
}

/*
34. 在排序数组中查找元素的第一个和最后一个位置
执行用时：4 ms, 在所有 Go 提交中击败了95.92%的用户
内存消耗：3.8 MB, 在所有 Go 提交中击败了100.00%的用户
通过测试用例：88 / 88
*/
func SearchRange(nums []int, target int) []int {
	if len(nums) == 0 {
		return []int{-1, -1}
	}
	left := 0
	right := len(nums) - 1
	resl := -1
	resr := -1
	for left < right {
		if nums[(left+right)/2] >= target {
			right = (left + right) / 2
		} else {
			left = (left+right)/2 + 1
		}
	}
	if nums[right] != target {
		return []int{-1, -1}
	}
	resl = right
	left = right
	right = len(nums) - 1
	for left < right {
		if nums[(left+right+1)/2] <= target {
			left = (left + right + 1) / 2
		} else {
			right = (left+right+1)/2 - 1
		}
	}
	resr = right
	return []int{resl, resr}
}

/*
35. 搜索插入位置
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.8 MB, 在所有 Go 提交中击败了57.05%的用户
通过测试用例：64 / 64
*/
func SearchInsert(nums []int, target int) int {
	n := len(nums)
	if n == 1 {
		if nums[0] < target {
			return 1
		} else {
			return 0
		}
	}

	l := 0
	r := n - 1
	for l < r {
		if l+1 == r {
			if nums[l] >= target {
				return l
			} else if nums[r] >= target {
				return r
			} else {
				return r + 1
			}
		}
		mid := (l + r) / 2
		if nums[mid] == target {
			return mid
		}
		if nums[mid] > target {
			r = mid
		} else {
			l = mid
		}
	}
	return 0
}

/*
36. 有效的数独
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.5 MB, 在所有 Go 提交中击败了73.33%的用户
通过测试用例：507 / 507
*/
func IsValidSudoku(board [][]byte) bool {
	var rows, columns [9][9]int
	var subboxes [3][3][9]int

	for i, row := range board {
		for j, v := range row {
			if v == '.' {
				continue
			}
			index := v - '1'
			rows[i][index]++
			columns[j][index]++
			subboxes[i/3][j/3][index]++
			if rows[i][index] > 1 || columns[j][index] > 1 || subboxes[i/3][j/3][index] > 1 {
				return false
			}
		}
	}

	return true
}

/*
37. 解数独
状态压缩
1. 使用 bitset<9> 来压缩存储每一行、每一列、每一个 3x3 宫格中 1-9 是否出现
2. 这样每一个格子就可以计算出所有不能填的数字，然后得到所有能填的数字 getPossibleStatus()
3. 填入数字和回溯时，只需要更新存储信息
4. 每个格子在使用时，会根据存储信息重新计算能填的数字
回溯
1. 每次都使用 getNext() 选择能填的数字最少的格子开始填，这样填错的概率最小，回溯次数也会变少
2. 使用 fillNum() 在填入和回溯时负责更新存储信息
3. 一旦全部填写成功，一路返回 true ，结束递归
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.6 MB, 在所有 Go 提交中击败了5.88%的用户
通过测试用例：6 / 6
*/
var rows, columns [9][9]int
var subboxes [3][3][9]int

func SolveSudoku(board [][]byte) {
	count := 0
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			rows[i][j] = 0
			columns[i][j] = 0
			subboxes[i/3][j/3] = [9]int{0, 0, 0, 0, 0, 0, 0, 0, 0}
		}
	}
	for i, row := range board {
		for j, v := range row {
			if v == '.' {
				count++
				continue
			}
			index := v - '1'
			rows[i][index]++
			columns[j][index]++
			subboxes[i/3][j/3][index]++
		}
	}
	dfs37(board, count)
}
func getPossibleStatus(i, j int) (res []int) {
	for k := 0; k < 9; k++ {
		if rows[i][k] == 0 && columns[j][k] == 0 && subboxes[i/3][j/3][k] == 0 {
			res = append(res, k)
		}
	}
	return
}

type next37 struct {
	I int `json:"i"`
	J int `json:"j"`
}

func getNext(board [][]byte) next37 {
	min := 9
	res := next37{}
	for i, row := range board {
		for j, _ := range row {
			if board[i][j] != '.' {
				continue
			}
			tmp := getPossibleStatus(i, j)
			if len(tmp) == 0 {
				return next37{
					I: -1,
					J: -1,
				}
			}
			if len(tmp) < min {
				min = len(tmp)
				res.I = i
				res.J = j
			}
		}
	}
	return res
}
func fillNum(i, j, n int, b bool) {
	if b {
		rows[i][n]++
		columns[j][n]++
		subboxes[i/3][j/3][n]++
	} else {
		rows[i][n]--
		columns[j][n]--
		subboxes[i/3][j/3][n]--
	}
}
func dfs37(board [][]byte, count int) bool {
	//待填数量0，成功
	if count == 0 {
		return true
	}
	next := getNext(board)
	//存在不能填入的格子
	if next.I == -1 {
		return false
	}
	acceptNums := getPossibleStatus(next.I, next.J)
	//存在不能填入的格子
	if len(acceptNums) == 0 {
		return false
	}
	for _, v := range acceptNums {
		fillNum(next.I, next.J, v, true)
		board[next.I][next.J] = byte(v) + '1'
		if dfs37(board, count-1) {
			return true
		}
		//失败回退
		fillNum(next.I, next.J, v, false)
		board[next.I][next.J] = '.'
	}
	return false
}

/*
38. 外观数列
执行用时：4 ms, 在所有 Go 提交中击败了59.55%的用户
内存消耗：7 MB, 在所有 Go 提交中击败了35.47%的用户
通过测试用例：30 / 30
*/
func CountAndSay(n int) string {
	if n == 1 {
		return "1"
	}
	s := CountAndSay(n - 1)
	fmt.Println("s:", s)
	res := ""
	for i := 0; i < len(s); {
		j := i
		for j < len(s) && s[j] == s[i] {
			j++
		}
		if j < len(s) {
			res += strconv.Itoa(j-i) + string(s[i])
		} else {
			if i == len(s)-1 {
				res += "1" + string(s[i])
				return res
			} else {
				res += strconv.Itoa(j-i) + string(s[i])
				return res
			}
		}

		i = j
	}
	return res
}

/*
39. 组合总和
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.9 MB, 在所有 Go 提交中击败了54.60%的用户
通过测试用例：171 / 171
*/
func CombinationSum(candidates []int, target int) [][]int {
	res := [][]int{}
	tmp := []int{}
	var dfs39 func(target int, index int)
	dfs39 = func(target int, index int) {
		if index == len(candidates) {
			return
		}
		if target == 0 {
			res = append(res, append([]int{}, tmp...))
			return
		}
		//跳过index
		dfs39(target, index+1)
		// 选择index
		if target-candidates[index] >= 0 {
			tmp = append(tmp, candidates[index])
			dfs39(target-candidates[index], index)
			//tmp还原，减去前面append添加的最新一位
			tmp = tmp[:len(tmp)-1]
		}
	}
	dfs39(target, 0)
	return res
}

/*
40. 组合总和 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.4 MB, 在所有 Go 提交中击败了53.63%的用户
通过测试用例：176 / 176
*/
func CombinationSum2(candidates []int, target int) [][]int {
	sort.Ints(candidates)
	//m记录candidates每个值出现的次数，0记录值，1记录次数
	//后续回溯m替代candidates
	var m [][2]int
	for _, v := range candidates {
		if m == nil || v != m[len(m)-1][0] {
			m = append(m, [2]int{v, 1})
		} else {
			m[len(m)-1][1]++
		}
	}
	res := [][]int{}
	tmp := []int{}
	var dfs39 func(target int, index int)
	dfs39 = func(target int, index int) {
		if target == 0 {
			res = append(res, append([]int{}, tmp...))
			return
		}
		if index == len(m) || m[index][0] > target {
			return
		}
		//跳过index
		dfs39(target, index+1)
		//选择index,1-n次
		n := m[index][1]
		if m[index][1] > target/m[index][0] {
			n = target / m[index][0]
		}

		//相同的处理，选1-n次
		for i := 1; i <= n; i++ {
			tmp = append(tmp, m[index][0])
			dfs39(target-(m[index][0]*i), index+1)
		}

		//tmp还原，减去前面append添加的最新n位
		tmp = tmp[:len(tmp)-n]
	}
	dfs39(target, 0)
	return res
}

/*
41. 缺失的第一个正数
对于遍历到的数 x，如果它在[1,len(nums)]的范围内，那么就将数组中的第 x−1 个位置（注意：数组下标从 00 开始）打上「标记」。
在遍历结束之后，如果所有的位置都被打上了标记，那么答案是 len(nums)+1，否则答案是最小的没有打上标记的位置加1。
不在[1,len(nums)] 范围内的数修改成任意一个大于 len(nums) 的数（例如 len(nums)+1）。
这样一来，数组中的所有数就都是正数了，因此我们就可以将「标记」表示为「负号」
负数变为len(nums)+1；value<=len(nums)元素对应的nums[value-1]变负；返回第一个>0的下标+1
执行用时：112 ms, 在所有 Go 提交中击败了33.21%的用户
内存消耗：24.5 MB, 在所有 Go 提交中击败了99.94%的用户
通过测试用例：173 / 173
*/
func FirstMissingPositive(nums []int) int {
	n := len(nums)
	for i := 0; i < n; i++ {
		if nums[i] <= 0 {
			nums[i] = n + 1
		}

	}
	fmt.Println(nums)
	for i := 0; i < n; i++ {
		tmp := int(math.Abs(float64(nums[i])))
		if tmp <= n {
			nums[tmp-1] = -int(math.Abs(float64(nums[tmp-1])))
		}
	}
	fmt.Println(nums)
	for i := 0; i < n; i++ {
		if nums[i] > 0 {
			return i + 1
		}
	}
	return n + 1
}

/*
42. 接雨水
双指针，如果一端有更高的条形块（例如右端），积水的高度依赖于当前方向的高度（从左到右）
左指针<左max，面积增加，否则左max更新；右指针同理
每步选择左右指针中值更小的更新
执行用时：12 ms, 在所有 Go 提交中击败了28.89%的用户
内存消耗：5.1 MB, 在所有 Go 提交中击败了62.42%的用户
通过测试用例：322 / 322
*/
func Trap(height []int) int {
	res := 0
	l := 0
	r := len(height) - 1
	lmax := 0
	rmax := 0
	for l < r {
		if height[l] < height[r] {
			if height[l] > lmax {
				lmax = height[l]
			} else {
				res += lmax - height[l]
			}
			l++
		} else {
			if height[r] > rmax {
				rmax = height[r]
			} else {
				res += rmax - height[r]
			}
			r--
		}
	}
	return res
}

/*
43. 字符串相乘
num1[i] x num2[j] 的结果为 tmp(位数为两位，"0x","xy"的形式)，其第一位位于 res[i+j]，第二位位于 res[i+j+1]。
执行用时：12 ms, 在所有 Go 提交中击败了28.36%的用户
内存消耗：4.9 MB, 在所有 Go 提交中击败了31.55%的用户
通过测试用例：311 / 311
*/
func Multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	l1 := len(num1)
	l2 := len(num2)
	tmp := make([]int, l1+l2)
	for i := 0; i < l1; i++ {
		for j := 0; j < l2; j++ {
			n1, _ := strconv.Atoi(string(num1[i]))
			n2, _ := strconv.Atoi(string(num2[j]))
			tmp[i+j] += n1 * n2 / 10
			tmp[i+j+1] += n1 * n2 % 10
		}
		fmt.Println(tmp)
	}
	for i := l1 + l2 - 1; i > 0; i-- {
		if tmp[i] >= 10 {
			tmp[i-1] += tmp[i] / 10
			tmp[i] = tmp[i] % 10
		}
	}
	res := ""
	zero := true
	for _, v := range tmp {
		if v != 0 {
			zero = false
		}
		if !zero {
			res += strconv.Itoa(v)
		}
	}
	return res
}

/*
44. 通配符匹配
dp[i][j] =
1. s[i] == p[j]: dp[i-1][j-1]
2. p[j] == '?': dp[i-1][j-1]
3. p[j] == '*': dp[i-1][j] || dp[i][j-1]
执行用时：12 ms, 在所有 Go 提交中击败了67.78%的用户
内存消耗：6.3 MB, 在所有 Go 提交中击败了77.82%的用户
通过测试用例：1811 / 1811
*/
func IsMatch44(s string, p string) bool {
	m, n := len(s), len(p)
	f := make([][]bool, m+1)
	for i := 0; i < len(f); i++ {
		f[i] = make([]bool, n+1)
	}
	f[0][0] = true
	for i := 1; i <= n; i++ {
		if p[i-1] == '*' {
			f[0][i] = true
		} else {
			break
		}
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == '*' {
				f[i][j] = f[i][j-1] || f[i-1][j]
			} else if p[j-1] == '?' || s[i-1] == p[j-1] {
				f[i][j] = f[i-1][j-1]
			}
		}
	}
	return f[m][n]
}

/*
45. 跳跃游戏 II
执行用时：12 ms, 在所有 Go 提交中击败了72.05%的用户
内存消耗：5.8 MB, 在所有 Go 提交中击败了99.40%的用户
通过测试用例：109 / 109
*/
func Jump(nums []int) int {
	n := len(nums)
	step := 0
	//每次最远可达范围
	far := 0
	//每次右边界
	r := 0
	//每次在当前范围中选一个最远的
	for i := 0; i < n-1; i++ {
		if i+nums[i] > far {
			far = i + nums[i]
		}
		if i == r {
			r = far
			step++
		}
	}
	return step
}

/*
46. 全排列
回溯，每次一个，将已填入的和每次的first交换，可以确保第i个之后都是未填入的
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.5 MB, 在所有 Go 提交中击败了85.30%的用户
通过测试用例：26 / 26
*/
func Permute(nums []int) [][]int {
	res := [][]int{}
	backtrack46(nums, 0, &res)
	return res
}
func backtrack46(nums []int, first int, res *[][]int) {
	if first == len(nums) {
		//fmt.Println("append nums:", append([]int{}, nums...))
		//直接append nums顺序会有问题，每次需要构造新的[]int{}
		*res = append(*res, append([]int{}, nums...))
		//fmt.Println("append res:", *res)
		return
	}
	for i := first; i < len(nums); i++ {
		tmp := nums[i]
		nums[i] = nums[first]
		nums[first] = tmp
		backtrack46(nums, first+1, res)
		tmp = nums[i]
		nums[i] = nums[first]
		nums[first] = tmp
	}
	return
}

/*
47. 全排列 II
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：3.6 MB, 在所有 Go 提交中击败了87.13%的用户
通过测试用例：33 / 33
*/
func PermuteUnique(nums []int) [][]int {
	sort.Ints(nums)
	n := len(nums)
	tmp := []int{}
	res := [][]int{}
	//记录是否使用
	vis := make([]bool, n)
	var backtrack func(int)
	backtrack = func(index int) {
		if index == n {
			res = append(res, append([]int(nil), tmp...))
			return
		}
		for i, v := range nums {
			//nums[i]已使用,或者i-1与i相同且未使用
			if vis[i] || i > 0 && !vis[i-1] && v == nums[i-1] {
				continue
			}
			tmp = append(tmp, v)
			vis[i] = true
			backtrack(index + 1)
			vis[i] = false
			tmp = tmp[:len(tmp)-1]
		}
	}
	backtrack(0)
	return res
}

/*
48. 旋转图像
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：2.1 MB, 在所有 Go 提交中击败了60.49%的用户
通过测试用例：21 / 21
*/
func Rotate(matrix [][]int) {
	//m[i][j]=>m[j][n-1-i] => m[i][j]=>m[j][i]=>m[j][n-1-i]
	//顺时针转90°=>主对角线旋转+上下反转
	//先左右再对角线代码好写
	n := len(matrix)
	for i := 0; i < n/2; i++ {
		matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
	}
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}

/*
49. 字母异位词分组
执行用时：20 ms, 在所有 Go 提交中击败了80.74%的用户
内存消耗：9.2 MB, 在所有 Go 提交中击败了5.80%的用户
通过测试用例：117 / 117
*/
func GroupAnagrams(strs []string) [][]string {
	m := make(map[string][]string)
	for _, v := range strs {
		//每个字符串排序，相同则记录在map中排序后的key的value
		tmp := []byte(v)
		sort.Slice(tmp, func(i, j int) bool { return tmp[i] < tmp[j] })
		sortedStr := string(tmp)
		m[sortedStr] = append(m[sortedStr], v)
	}
	var res [][]string
	for _, v := range m {
		res = append(res, v)
	}
	return res
}

/*
50. Pow(x, n)
快速幂，O(logn)，每次x翻倍n减半
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.9 MB, 在所有 Go 提交中击败了99.91%的用户
通过测试用例：305 / 305
*/
func MyPow(x float64, n int) float64 {
	if n == 0 {
		return 1.0
	}
	//b记录是否需要1/
	b := false
	if n < 0 {
		b = true
		n = -n
	}
	res := x
	if n == 1 {
		res = x
	} else {
		//奇数需要额外*x，偶数不需要
		if n%2 == 0 {
			res = MyPow(x*x, n/2)
		} else {
			res = MyPow(x*x, n/2) * x
		}

	}
	if b {
		return 1 / res
	}
	return res
}
