# [2, 1, 2]	1	6 # 1 1 2 > 1**2 1**2 2**2 > 6
# [4, 3, 3]	4	12 # 2 2 2 > 2**2 2**2 2**2 > 12
# [1,1]	3	0
def solution(works, n):
    if sum(works) <= n:
        return 0
    
    while (n > 0):
        works.sort(reverse=True)
        n -=1                     
        works[0] -=1
        if n == 0:
            break
    res = 0
    for work in works:
        res += work**2
            
    return res


print(solution([2,1,2], 1))
print(solution([4, 3, 3],	4))
print(solution([1,1],	3))

