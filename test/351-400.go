package test

import (
	"fmt"
	"math"
	"sort"
)

/*
todo 会员：351,353,356,358-362,364
*/

/*
352. 将数据流变为多个不相交区间
*/

/*
354. 俄罗斯套娃信封问题
执行用时：162 ms, 在所有 Go 提交中击败了26.62%的用户
内存消耗：21.81 MB, 在所有 Go 提交中击败了12.33%的用户
*/
func MaxEnvelopes(envelopes [][]int) int {
	//二维最长递增子序列，先weight增序，weight相同height降序，找height的最长递增子序列
	sort.Slice(envelopes, func(i, j int) bool {
		return envelopes[i][0] < envelopes[j][0] || (envelopes[i][0] == envelopes[j][0] && envelopes[i][1] > envelopes[j][1])
	})

	fmt.Println(envelopes)

	var temp []int
	for _, v := range envelopes {
		temp = append(temp, v[1])
	}

	fmt.Println(temp)

	l := len(temp)
	if l == 0 {
		return 0
	}

	/*
		设当前已求出的最长上升子序列的长度为 len（初始时为 1），从前往后遍历数组 nums，在遍历到 nums[i] 时：
		如果 nums[i]>d[len] ，则直接加入到 d 数组末尾，并更新 len=len+1；
		否则，在 d 数组中二分查找，找到第一个比 nums[i]小的数 d[k]，并更新 d[k+1]=nums[i]。
	*/
	var res []int
	for i := 0; i < l; i++ {
		t := sort.SearchInts(res, temp[i])
		if t == len(res) {
			res = append(res, temp[i])
		} else {
			res[t] = temp[i]
		}
	}

	return len(res)
}

/*
355. 设计推特 todo
*/

/*
357. 统计各位数字都不同的数字个数
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.80 MB, 在所有 Go 提交中击败了80.00%的用户
*/
func CountNumbersWithUniqueDigits(n int) int {
	if n == 0 {
		return 1
	}
	/*
		f(0)=1
		f(1)=10
		f(2)=9*9+f(1)
		f(3)=9*9*8+f(2)
		f(4)=9*9*8*7+f(3)
	*/
	res := 10
	last := 9
	for i := 2; i <= n; i++ {
		tmp := last * (10 - i + 1)
		res += tmp
		last = tmp
	}
	return res
}

/*
363. 矩形区域不超过 K 的最大数值和
执行用时：173 ms, 在所有 Go 提交中击败了70.00%的用户
内存消耗：4.85 MB, 在所有 Go 提交中击败了75.00%的用户
*/
func MaxSumSubmatrix(matrix [][]int, k int) int {
	//todo 非暴力方法解题
	r, c := len(matrix), len(matrix[0])
	tmp := make([][]int, r)
	for i := 0; i < r; i++ {
		tmp[i] = make([]int, c)
	}
	/*
		1 0  1
		0 -2 3

		1 1 2
		1 -1 3
	*/

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if j > 0 && i > 0 {
				tmp[i][j] = matrix[i][j] + tmp[i][j-1] + tmp[i-1][j] - tmp[i-1][j-1]
			} else if i > 0 && j == 0 {
				tmp[i][j] = matrix[i][j] + tmp[i-1][j]
			} else if i == 0 && j > 0 {
				tmp[i][j] = matrix[i][j] + tmp[i][j-1]
			} else {
				tmp[i][j] = matrix[i][j]
			}
			if tmp[i][j] == k {
				return k
			}
		}
	}

	maxValue := math.MinInt
	//fmt.Println(tmp)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			for m := i; m < r; m++ {
				for n := j; n < c; n++ {
					var area int
					if i > 0 && j > 0 {
						area = max(maxValue, tmp[m][n]-tmp[i-1][n]-tmp[m][j-1]+tmp[i-1][j-1])
					} else if i > 0 && j == 0 {
						area = max(maxValue, tmp[m][n]-tmp[i-1][n])
					} else if i == 0 && j > 0 {
						area = max(maxValue, tmp[m][n]-tmp[m][j-1])
					} else {
						area = max(maxValue, tmp[m][n])
					}
					if area == k {
						return k
					}
					if area < k {
						maxValue = max(area, maxValue)
					}

				}
			}
		}
	}

	return maxValue
}

/*
365. 水壶问题
//数学
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.80 MB, 在所有 Go 提交中击败了99.84%的用户
//dfs
执行用时：1597 ms, 在所有 Go 提交中击败了5.06%的用户
内存消耗：583.55 MB, 在所有 Go 提交中击败了5.07%的用户
*/
func CanMeasureWater(jug1Capacity int, jug2Capacity int, targetCapacity int) bool {

	if jug1Capacity+jug2Capacity < targetCapacity {
		return false
	}
	if jug1Capacity == 0 || jug2Capacity == 0 {
		return targetCapacity == 0 || jug1Capacity+jug2Capacity == targetCapacity
	}

	//数学方法
	//贝祖定理告诉我们，ax+by=z 有解当且仅当 z 是 x, y 的最大公约数的倍数。
	//因此我们只需要找到 x, y 的最大公约数并判断 z 是否是它的倍数即可
	/*var gcd func(a, b int) int
	gcd = func(a, b int) int {
		for a%b != 0 {
			a, b = b, a%b
		}
		return b
	}

	return targetCapacity%gcd(jug1Capacity, jug2Capacity) == 0*/

	//或dfs，记录两个状态不出现多次，6种操作

	m := make(map[int]map[int]bool)
	for i := 0; i <= jug1Capacity; i++ {
		m[i] = make(map[int]bool)
	}
	var dfs func(a, b int) bool
	dfs = func(a, b int) bool {
		if a > jug1Capacity {
			a = jug1Capacity
		}
		if b > jug2Capacity {
			b = jug2Capacity
		}
		_, ok := m[a][b]
		if ok {
			return false
		}
		if a+b == targetCapacity {
			return true
		}
		m[a][b] = false
		r := dfs(a, 0)
		if r {
			return r
		}
		r = dfs(0, b)
		if r {
			return r
		}
		r = dfs(a, jug2Capacity)
		if r {
			return r
		}
		r = dfs(jug1Capacity, b)
		if r {
			return r
		}
		if a-jug2Capacity+b > 0 {
			r = dfs(a-jug2Capacity+b, jug2Capacity)
		} else {
			r = dfs(0, a+b)
		}
		if r {
			return r
		}
		if b-jug1Capacity+a > 0 {
			r = dfs(jug1Capacity, b-jug1Capacity+a)
		} else {
			r = dfs(a+b, 0)
		}
		if r {
			return r
		}
		return false
	}

	return dfs(0, 0)
}
