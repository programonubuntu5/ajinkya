
print("A  B  Output")

w1 = 1
w2 = -1
threshold = 1

for A in [0, 1]:
    for B in [0, 1]:
        net = A * w1 + B * w2

        if net >= threshold:
            output = 1
        else:
            output = 0

        print(A, B, output)
