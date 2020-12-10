import numpy as np

G1 = np.array([[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 2, -16, 2, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]])

G2 = np.array([[1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, -24, 1, 1],
             [1, 1, 2, 1, 1],
             [1, 1, 1, 1, 1]])

u = np.arange(-2,3)
v = np.arange(-2,3)
uu, vv = np.meshgrid(u,v)

fft_H = uu**2 + vv**2
H = np.fft.ifftshift(np.fft.ifft2(fft_H.astype(np.float64)))

k = np.arange(-5,5,0.1)
error1 = []
error2 = []
for i in range(len(k)):
    error1.append(np.linalg.norm(G1-k[i]*H,2))
    error2.append(np.linalg.norm(G2-k[i]*H,2))

print("Optimal value of k in case (a) is: %.4f"%(k[np.argmin(error1)]))
print("Optimal value of k in case (b) is: %.4f"%(k[np.argmin(error2)]))