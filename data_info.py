import json

length_of_q = 0
value_of_r = 0
num_values = 0

with open("train(smaller).json") as f:
    for line in f:
        num_values += 1
        pair = json.loads(line.strip())
        q, r = pair["question"], pair["solution"]
        length_of_q += len(q.split(" "))
        value_of_r += r
    print("Average Input Length (# of words):", length_of_q/num_values)
    print("Average Output Value:", value_of_r/num_values)

length_of_q = 0
value_of_r = 0
num_values = 0

with open("valid(smaller).json") as f:
    for line in f:
        num_values += 1
        pair = json.loads(line.strip())
        q, r = pair["question"], pair["solution"]
        length_of_q += len(q.split(" "))
        value_of_r += r
    print("Average Input Length (# of words):", length_of_q/num_values)
    print("Average Output Value:", value_of_r/num_values)

length_of_q = 0
value_of_r = 0
num_values = 0

with open("test(smaller).json") as f:
    for line in f:
        num_values += 1
        pair = json.loads(line.strip())
        q, r = pair["question"], pair["solution"]
        length_of_q += len(q.split(" "))
        value_of_r += r
    print("Average Input Length (# of words):", length_of_q/num_values)
    print("Average Output Value:", value_of_r/num_values)
    
