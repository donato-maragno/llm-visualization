"""Original LeetCode-style Python exercises with explicit, deterministic tests."""

import json
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    args: list[JsonValue] = Field(max_length=8)
    expected: JsonValue


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9-]{1,80}$")
    title: str = Field(min_length=1, max_length=120)
    category: Literal["arrays", "strings", "hashing", "stack", "dynamic-programming", "search"] = "arrays"
    description: str = Field(min_length=10, max_length=6000)
    tests: list[TestCase] = Field(min_length=1, max_length=30)
    normalize: Literal["exact", "unordered"] = "exact"

    @field_validator("tests")
    @classmethod
    def small_inputs(cls, tests):
        if len(json.dumps([t.model_dump() for t in tests], allow_nan=False)) > 65536:
            raise ValueError("Test data is limited to 64 KB per problem.")
        return tests

    def prompt(self):
        examples = json.dumps([t.model_dump() for t in self.tests[:2]], ensure_ascii=False)
        return (
            "Solve this algorithm problem in Python 3. Return only Python code, "
            "with a top-level function named solve. No input(), file/network access, "
            "prints, example invocations, or test harness. Standard algorithm imports "
            "from typing, collections, math, heapq, bisect, itertools and functools are allowed.\n\n"
            f"{self.title}\n{self.description}\n\n"
            f"Examples (args are positional arguments to solve; expected is its return value):\n{examples}\n"
            "Return the complete implementation of solve."
        )


def problem(id, title, category, description, cases, normalize="exact"):
    return Problem(id=id, title=title, category=category, description=description,
                   tests=[TestCase(args=args, expected=expected) for args, expected in cases], normalize=normalize)


CATALOG = [
    problem("two-sum", "Two Sum", "hashing", "Implement solve(nums, target). Return the two distinct zero-based indices whose values sum to target. Each input has exactly one pair. Index order does not matter.", [
        (([2,7,11,15],9),[0,1]), (([3,2,4],6),[1,2]), (([3,3],6),[0,1]), (([-3,4,3,90],0),[0,2])], "unordered"),
    problem("valid-parentheses", "Valid Parentheses", "stack", "Implement solve(s). Return True if every bracket in s is correctly matched and nested. Only ()[]{} occur. The empty string is valid.", [
        (("()[]{}",),True), (("([)]",),False), (("{[]}",),True), (("(",),False), (("",),True)]),
    problem("common-prefix", "Longest Common Prefix", "strings", "Implement solve(words). Return the longest shared prefix of all strings, or the empty string if words is empty or no prefix exists.", [
        ((["flower","flow","flight"],),"fl"), ((["dog","racecar","car"],),""), (([],),""), ((["solo"],),"solo"), ((["","a"],),"")]),
    problem("contains-duplicate", "Contains Duplicate", "hashing", "Implement solve(nums). Return True if any integer occurs at least twice; otherwise return False.", [
        (([1,2,3,1],),True), (([1,2,3,4],),False), (([],),False), (([0,0],),True)]),
    problem("maximum-subarray", "Maximum Subarray", "dynamic-programming", "Implement solve(nums). Given a nonempty integer list, return the largest sum of a nonempty contiguous subarray.", [
        (([-2,1,-3,4,-1,2,1,-5,4],),6), (([-3,-2,-5],),-2), (([1],),1), (([5,4,-1,7,8],),23)]),
    problem("binary-search", "Binary Search", "search", "Implement solve(nums, target). nums is strictly increasing. Return target's zero-based index, or -1 if absent.", [
        (([-1,0,3,5,9,12],9),4), (([-1,0,3,5,9,12],2),-1), (([],1),-1), (([5],5),0)]),
    problem("merge-intervals", "Merge Overlapping Intervals", "arrays", "Implement solve(intervals). Each item is [start, end] with start <= end. Merge overlapping or touching closed intervals and return them in increasing start order. Empty input returns [].", [
        (([[1,3],[2,6],[8,10],[15,18]],),[[1,6],[8,10],[15,18]]), (([[1,4],[4,5]],),[[1,5]]), (([],),[]), (([[5,7],[1,2],[2,4]],),[[1,4],[5,7]])]),
    problem("move-zeroes", "Move Zeroes", "arrays", "Implement solve(nums). Return a list with all zeroes moved to the end while preserving the order of nonzero values. You may modify nums but must RETURN the resulting list.", [
        (([0,1,0,3,12],),[1,3,12,0,0]), (([0],),[0]), (([],),[]), (([1,2],),[1,2])]),
    problem("climbing-stairs", "Climbing Stairs", "dynamic-programming", "Implement solve(n). For 0 <= n <= 35, return the number of ways to climb n stairs using steps of size 1 or 2. There is one way to climb zero stairs.", [
        ((0,),1), ((1,),1), ((2,),2), ((5,),8), ((10,),89)]),
    problem("longest-substring", "Longest Substring Without Repeats", "strings", "Implement solve(s). Return the length of the longest contiguous substring containing no repeated characters.", [
        (("abcabcbb",),3), (("bbbbb",),1), (("pwwkew",),3), (("",),0), (("abba",),2)]),
    problem("product-except-self", "Product Except Self", "arrays", "Implement solve(nums). Return a list where result[i] is the product of every input element except nums[i]. Use O(n) time and no division. Input has at least two elements.", [
        (([1,2,3,4],),[24,12,8,6]), (([-1,1,0,-3,3],),[0,0,9,0,0]), (([0,0],),[0,0]), (([2,3],),[3,2])]),
    problem("single-number", "Single Number", "hashing", "Implement solve(nums). Every integer appears exactly twice except one that appears once. Return that integer. Input is nonempty.", [
        (([2,2,1],),1), (([4,1,2,1,2],),4), (([-1],),-1)]),
    problem("stock-profit", "Best Single Stock Trade", "arrays", "Implement solve(prices). Return the maximum profit from buying once and selling on a later day. Return 0 if no profitable trade exists.", [
        (([7,1,5,3,6,4],),5), (([7,6,4,3,1],),0), (([],),0), (([1,2],),1)]),
    problem("valid-palindrome", "Valid Alphanumeric Palindrome", "strings", "Implement solve(s). Ignore non-alphanumeric ASCII characters and letter case. Return whether the remaining string is a palindrome. Empty cleaned strings are palindromes.", [
        (("A man, a plan, a canal: Panama",),True), (("race a car",),False), ((" ",),True), (("0P",),False)]),
    problem("valid-anagram", "Valid Anagram", "hashing", "Implement solve(a, b). The inputs contain lowercase English letters. Return whether b is a rearrangement of a with exactly the same multiplicities.", [
        (("anagram","nagaram"),True), (("rat","car"),False), (("a","aa"),False), (("",""),True)]),
    problem("search-insert", "Search Insert Position", "search", "Implement solve(nums, target). nums is sorted and contains distinct integers. Return target's index if present, otherwise the insertion index preserving order.", [
        (([1,3,5,6],5),2), (([1,3,5,6],2),1), (([1,3,5,6],7),4), (([],3),0)]),
    problem("majority-element", "Majority Element", "hashing", "Implement solve(nums). Return the integer occurring strictly more than len(nums)/2 times. Input is nonempty and a majority always exists.", [
        (([3,2,3],),3), (([2,2,1,1,1,2,2],),2), (([-1],),-1)]),
    problem("missing-number", "Missing Number", "arrays", "Implement solve(nums). The distinct values come from 0 through len(nums), inclusive, with exactly one number absent. Return the missing number.", [
        (([3,0,1],),2), (([0,1],),2), (([],),0), (([1],),0)]),
    problem("rotate-array", "Rotate Array Right", "arrays", "Implement solve(nums, k). Return nums rotated right by k positions, where k is nonnegative. Empty input returns []. You may modify nums but must return the resulting list.", [
        (([1,2,3,4,5,6,7],3),[5,6,7,1,2,3,4]), (([-1,-100,3,99],2),[3,99,-1,-100]), (([],10),[]), (([1,2],0),[1,2])]),
    problem("reverse-words", "Reverse Words", "strings", "Implement solve(s). Reverse the order of whitespace-separated words. Remove leading/trailing whitespace and use one space between output words.", [
        (("the sky is blue",),"blue is sky the"), (("  hello world  ",),"world hello"), (("a good   example",),"example good a"), (("",),"")]),
]


def get_problems(category="all"):
    return [p.model_copy(deep=True) for p in CATALOG if category == "all" or p.category == category]


def new_problem(existing, category="all"):
    available = [p for p in get_problems(category) if p.id not in existing]
    if not available:
        raise ValueError("No unused problems in this category. Choose another category or add a custom problem with tests.")
    return random.SystemRandom().choice(available)
