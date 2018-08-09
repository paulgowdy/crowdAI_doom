import matplotlib.pyplot as plt

with open('score_writer.txt', 'r') as f:

    z = f.readlines()

z = [float(x.strip()) for x in z]

print(len(z))
plt.figure()
plt.plot(z)
plt.show()
