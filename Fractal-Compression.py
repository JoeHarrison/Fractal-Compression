from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
%matplotlib inline

#Hyper parameters
mutation_rate = 0.05
crossover_rate = 0.5
gene_size = 256
population_size = 100
generations = 10000

fractal_iterations = 2
window_size_sqrt = 3
seed = 0

#Initalisation of rules
def init_windows():
    windows = np.zeros((gene_size,window_size_sqrt,window_size_sqrt))
    for i in range(gene_size):
        windows[i] = np.random.randint(255,size=(window_size_sqrt,window_size_sqrt))
    return windows
