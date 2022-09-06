class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for idx in range(len(nums)):
            for jdx in range((idx+1), len(nums)) :
                if (idx == jdx) :
                    continue
                if target == (nums[idx] + nums[jdx]):
                    return [idx, jdx]
        return [-1,-1]


a = Solution()
print(a.twoSum([2,7,11,15], 9))
print(a.twoSum([3,2,4], 6))