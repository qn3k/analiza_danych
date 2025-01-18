import numpy as np

def gauss_jordan(A, b):
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
    rows, cols = augmented_matrix.shape

    for i in range(rows):
        max_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
        if augmented_matrix[max_row, i] == 0:
            continue

        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        augmented_matrix[i] /= augmented_matrix[i, i]

        for j in range(rows):
            if j != i:
                augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]

    solution = augmented_matrix[:, -1]

    for i in range(rows):
        if np.allclose(augmented_matrix[i, :-1], 0) and not np.isclose(augmented_matrix[i, -1], 0):
            return "Brak rozwiązań"

    if any(np.allclose(augmented_matrix[i, :-1], 0) for i in range(rows)):
        return "Nieskończenie wiele rozwiązań"

    return np.round(solution, decimals=3)


A = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1],
    [1, 2, 1, 1, 1, 1],
    [1, 1, 2, 1, 1, 1],
    [1, 1, 1, 2, 1, 1]
], dtype=float)

b = np.array([1, 2, 2, 3, 4, 5], dtype=float)

# Wprowadzanie danych
#print("Podaj liczbę równań:")
#n = int(input())
n=6
'''print(f"Podaj współczynniki macierzy A ({n}x{n}):")
A = []
for i in range(n):
    row = list(map(float, input(f"Wiersz {i+1}: ").split()))
    A.append(row)
A = np.array(A, dtype=float)

print("Podaj wyrazy wolne wektora b:")
b = list(map(float, input().split()))
b = np.array(b, dtype=float)'''

# Rozwiązywanie układu równań
result = gauss_jordan(A, b)

if isinstance(result, str):
    print(result)
else:
    for i, val in enumerate(result):
        print(f"x{i+1} = {val}")
