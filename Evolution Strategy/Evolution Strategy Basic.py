"""
Find minimizer of Rosenbrock function (banana function) using Evolution Strategy  
https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt
from es_base import ES

DNA_SIZE = 2             # DNA (real number)
DNA_BOUND = [-50, 50]       # solution upper and lower bounds
N_GENERATIONS = 300
POP_SIZE = 300           # population size
N_KID = 100               # n kids per generation


def F(x): return (1-x[...,0])**2 + 100*(x[...,1]-x[...,0]**2)**2     # to find the minimizer of this function

es = ES(fitness=F, dna_length=DNA_SIZE, bound=DNA_BOUND, generations=N_GENERATIONS, population_size=POP_SIZE, offspring_size=N_KID, type='minimize')
pop = es.initialization(mean=0.0, std_dev=20.0)

plt.ion()       # something about plotting
x = np.linspace(*DNA_BOUND, 200)
y = np.linspace(*DNA_BOUND, 200)
X,Y = np.meshgrid(x, y)
Z = np.append(X[...,np.newaxis], Y[...,np.newaxis], axis=-1)
print(Z.shape)
Z = F(Z)
print(Z.shape)

plt.contourf(X, Y, Z, 8, alpha=.75, cmap=plt.cm.hot)
C = plt.contour(X, Y, Z, 8, colors='black', linewidth=.5)
plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())

for i in range(N_GENERATIONS):
    # something about plotting
    if 'sca' in globals(): sca.remove()
    if 'text' in globals(): text.remove()
    text = plt.text(x=DNA_BOUND[1]-15, y=DNA_BOUND[0]-5, s='iter: %d/%d' % (i+1,N_GENERATIONS), fontsize=12)
    sca = plt.scatter(x=pop['DNA'][...,0], y=pop['DNA'][...,1], s=200, lw=0, c='red', alpha=0.5)
    plt.pause(0.05)

    # ES part
    kids = es.get_offspring(pop)
    pop = es.put_kids(pop, kids)
    pop = es.selection(pop)
    best = pop['DNA'][-1]
    print('best: ', best, F(best))

plt.ioff(); plt.show()
