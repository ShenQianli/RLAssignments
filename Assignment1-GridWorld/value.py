import numpy as np

THRESHOLD = 1e-6
GAMMA = 0.9
MAX_ITER = 10000

v = np.zeros((5, 5))
eps = np.inf
iter_cnt = 0

delta_x, delta_y = np.array([-1, 0, 1, 0]), np.array([0, 1, 0, -1])

while eps > THRESHOLD and iter_cnt < MAX_ITER:
    q_prime = np.zeros((5, 5, 4))
    for x in range(5):
        for y in range(5):
            for a in range(4):
                x_prime = min(max(x + delta_x[a], 0), 4)
                y_prime = min(max(y + delta_y[a], 0), 4)
                r = 0
                if x == 0 and y == 1:
                    x_prime, y_prime = 4, 1
                    r = 10
                elif x == 0 and y == 3:
                    x_prime, y_prime = 2, 3
                    r = 5
                elif x_prime == x and y_prime == y:
                    r = -1
                q_prime[x, y, a] = r + GAMMA * v[x_prime, y_prime]
    v_prime = np.max(q_prime, axis=-1)
    eps = np.max(np.abs(v_prime - v))
    v = v_prime
    iter_cnt += 1

print(v)

