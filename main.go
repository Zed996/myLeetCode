package main

import (
	"fmt"
	"myLeetCode/test"
)

func main() {

	r := test.CanMeasureWater(3, 5, 4)
	//r := test.LengthOfLIS([]int{89, 53, 68, 45, 81})
	fmt.Println(r)
	//l1 := &test.ListNode{
	//	Val: 4,
	//	Next: &test.ListNode{
	//		Val: 2,
	//		Next: &test.ListNode{
	//			Val: 1,
	//			Next: &test.ListNode{
	//				Val: 3,
	//				Next: &test.ListNode{
	//					Val:  5,
	//					Next: nil,
	//				},
	//			},
	//		},
	//	},
	//}

	//s := []int{1, 2, 3, 0, 0, 0}
	//test.Merge88(s, 3, []int{2, 5, 6}, 3)
	//fmt.Println(s)
	//fmt.Println(i)

	//t := &test.TreeNode{
	//	Val: 1,
	//	Left: &test.TreeNode{
	//		Val: 2,
	//		Left: &test.TreeNode{
	//			Val:   4,
	//			Left:  nil,
	//			Right: nil,
	//		},
	//		Right: &test.TreeNode{
	//			Val:   5,
	//			Left:  nil,
	//			Right: nil,
	//		},
	//	},
	//	Right: &test.TreeNode{
	//		Val: 3,
	//		Left: &test.TreeNode{
	//			Val:   6,
	//			Left:  nil,
	//			Right: nil,
	//		},
	//		Right: nil,
	//	},
	//}
	//t := &test.TreeNode{
	//	Val: 3,
	//	Left: &test.TreeNode{
	//		Val:  1,
	//		Left: nil,
	//		Right: &test.TreeNode{
	//			Val:   2,
	//			Left:  nil,
	//			Right: nil,
	//		},
	//	},
	//	Right: &test.TreeNode{
	//		Val:   4,
	//		Left:  nil,
	//		Right: nil,
	//	},
	//}

	//t := test.ListNode{
	//	Val: 1,
	//	Next: &test.ListNode{
	//		Val: 2,
	//		Next: &test.ListNode{
	//			Val: 3,
	//			Next: &test.ListNode{
	//				Val: 4,
	//				Next: &test.ListNode{
	//					Val: 5,
	//					Next: &test.ListNode{
	//						Val:  6,
	//						Next: nil,
	//					},
	//				},
	//			},
	//		},
	//	},
	//}
	//s := []int{1, 1, 1, 2, 1}
	//i := test.IsSelfCrossing(s)
	//fmt.Println(i)
	//s := []string{"2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"}
	//s := []string{"abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"}
	//for _, v := range s {
	//	if test.IsNumber(v) {
	//		fmt.Println(v)
	//	}
	//}
	//	for i != nil {
	//		fmt.Println(i.Val)
	//		i = i.Next
	//	}
	/*
		1 0 1 0 0
		1 0 1 1 1
		1 1 1 1 1
		1 0 0 1 0
	*/
}
