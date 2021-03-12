inputs = [1,2,3]
w = [[0.5,0.1,0.2],[0.5,0.1,0.2],[0.5,0.1,0.2]]
b = [0.1,0.1,0.1]

outputs = list()
for i in range(3):
    outputs.append(sum([(inputs[j] * w[i][j]) for j in range(len(w[0]))]) + b[i])
    
print(outputs)
