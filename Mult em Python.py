import time

n = 1000

# matriz lista de lista26
a = [[1 for _ in range(n)] for _ in range(n)]
b = [[1 for _ in range(n)] for _ in range(n)]
c = [[0 for _ in range(n)] for _ in range(n)]
# i ou _ nao parece alterar o tempo

start = time.time()

for i in range(n):
    for j in range(n):
        for k in range(n):
            c[i][j] += a[i][k] * b[k][j]

end = time.time()

print(f"Tempo para multiplicar: {end - start} segundos")

