
def solution(triangle):
    n = len(triangle)
    dp = [[0] * (i + 1) for i in range(n)]
    dp[0][0] = triangle[0][0]

    for i in range(1, n):
        for j in range(i + 1):
            if j == 0: # L
                dp[i][j] = dp[i - 1][j] + triangle[i][j]
            elif j == i: # R
                dp[i][j] = dp[i - 1][j - 1] + triangle[i][j]
            # 중간 값 위에서 오는 최대값
            else:
                dp[i][j] = max(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]

    print(dp)
    return max(dp[-1])  # max

res = solution([[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]])
print(res)