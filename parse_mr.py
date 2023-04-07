def fix_pattern(line, pattern):
    mod = False
    if pattern in line:
        line2 = line.split(pattern)
        line = pattern[1:].join(line2)
        mod = True
    return mod, line

with open('movieReviews.txt', 'r') as fin:
    lines = fin.readlines()

lines2 = []

modified = 0
for line in lines:
    mod = False
    mod2, line = fix_pattern(line, " n't")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " 's")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " 'd")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " 're")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " 'll")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " ,")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " .")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " ?")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " !")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " :")
    if mod2: mod = True
    mod2, line = fix_pattern(line, " ;")
    if mod2: mod = True
    mod2, line = fix_pattern(line, "\/")
    if mod2: mod = True
    if mod: 
        modified += 1
    else:
        print(line)
    lines2.append(line)

print(len(lines))
print(len(lines2))
print(modified)

with open('movieReviews2.txt', 'w') as fout:
    for line in lines2:
        print(line, file=fout, end="")