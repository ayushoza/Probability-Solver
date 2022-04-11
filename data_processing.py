f_in = "arithmetic__add_or_sub.txt"

train_out = "train.json"
valid_out = "valid.json"
test_out = "test.json"

f_in = open(f_in, 'r')

train_out = open(train_out, 'a')
valid_out = open(valid_out, 'a')
test_out = open(test_out, 'a')

lines = f_in.readlines()
length = len(lines)/2
train = int(length*0.6)
valid = int(length*0.2)
test = int(length*0.2)

for i in range(0, len(lines), 2):
    qs = lines[i].rstrip()
    line = '{"question": "' + qs + '", "solution": ' + \
        str(eval(lines[i+1])) + '}\n'
    if train > 0:
        train -= 1
        to_write = train_out
    elif valid > 0:
        valid -= 1
        to_write = valid_out
    else:
        test -= 1
        to_write = test_out
    to_write.writelines([line])

f_in.close()
train_out.close()
valid_out.close()
test_out.close()
