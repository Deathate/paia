car_info = [[9, 89, 2046], [8, 24, 2100], [8, 25, 133], [8, 25, 1470], [7, 1, 156]]
user_score = []
for car in car_info:
    score = 10000 * car[0] - 10 * car[1] - 0.001 * car[2]
    user_score.append(score)
    print(car, score, "\n")
result = [user_score.index(x) for x in sorted(user_score, reverse=True)]
print(result)
