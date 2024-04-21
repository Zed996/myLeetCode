package test

import (
	"sort"
	"strconv"
)

/*
670. 最大交换
执行用时：0 ms, 在所有 Go 提交中击败了100.00%的用户
内存消耗：1.8 MB, 在所有 Go 提交中击败了83.19%的用户
通过测试用例：54 / 54
*/
func maximumSwap(num int) int {
	s := strconv.Itoa(num)
	b := []byte(s)
	sort.Slice(b, func(i, j int) bool {
		return b[i] > b[j]
	})

	for j := 0; j < len(s); j++ {
		if s[j] < b[j] {
			for k := len(s) - 1; k > j; k-- {
				if s[k] == b[j] {
					r := s[:j] + string(s[k]) + s[j+1:k] + string(s[j]) + s[k+1:]
					res, _ := strconv.Atoi(r)
					return res
				}
			}
		}
	}
	return num
}
