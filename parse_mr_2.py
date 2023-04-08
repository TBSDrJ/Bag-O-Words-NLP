# Every review has a rating 0 -> 4, so 5 different choices.
with open('movieReviews.txt', 'r') as fin:
    lines = fin.readlines()

counts = [0] * 5
for line in lines:
    rate_str = line[0]
    rate_int = int(line[0])
    filename = f"reviews/{rate_str}/{rate_str}_{counts[rate_int]:04}.txt"
    with open(filename, 'w') as fout:
        print(line, file=fout)
    counts[rate_int] += 1


