from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def decode(windows,fractal_iterations,seed):
    seed_matrix = np.array([[seed]])

    for i in range(fractal_iterations):
        size_y = seed_matrix.shape[0]
        size_x = seed_matrix.shape[1]
        new_matrix = np.zeros((size_y*window_size_sqrt,size_x*window_size_sqrt))

        for i in range(size_y):
            for j in range(size_x):
                seed_value = seed_matrix[i,j]
                new_matrix[i*window_size_sqrt : i*window_size_sqrt+window_size_sqrt, j*window_size_sqrt : j*window_size_sqrt+window_size_sqrt] = windows[int(seed_value)]
        seed_matrix = new_matrix

    return seed_matrix

def fitness(target_image,genetic_image):
    difference = np.subtract(target_image,genetic_image)
    abs = np.abs(difference)
    sum = np.sum(abs)

    return sum

def mutate(windows,mutation_rate,gene_size,window_size_sqrt):
    mutated_windows = np.zeros((gene_size,window_size_sqrt,window_size_sqrt))
    for i in range(gene_size):
        random_matrix = np.random.choice([-1,0,1],(window_size_sqrt,window_size_sqrt),[mutation_rate/2,1-mutation_rate,mutation_rate/2])
        mutated_windows[i] = np.clip(windows[i] + random_matrix,a_min=0,a_max=255)
    return mutated_windows

def crossover(windows1,windows2,crossover_rate):
    crossover_windows1 = np.zeros((gene_size,window_size_sqrt,window_size_sqrt))
    crossover_windows2 = np.zeros((gene_size,window_size_sqrt,window_size_sqrt))
    for i in range(gene_size):
        randi = np.random.choice([0,1],p=[1-crossover_rate,crossover_rate])
        if(randi):
            crossover_windows1[i] = windows2[i]
            crossover_windows2[i] = windows1[i]
        else:
            crossover_windows1[i] = windows1[i]
            crossover_windows2[i] = windows2[i]

    return crossover_windows1, crossover_windows2

if __name__ == '__main__':
    img = Image.open('test.png').convert("L")
    target = np.asarray(img)

    population = np.zeros((population_size,gene_size,window_size_sqrt,window_size_sqrt))

    for i in range(population_size):
        population[i] = init_windows()

    try:
        for i in range(generations):
            top = []
            for j in range(population_size):
                decoded = decode(population[j],fractal_iterations,seed)
                top.append((fitness(decoded,target),j))
            top = sorted(top, key=lambda tup: tup[0])

            population[0] = population[top[0][1]]

            for j in range(1,population_size - 3):
                population[j] = mutate(population[top[j][1]],mutation_rate,gene_size,window_size_sqrt)

            for j in range(1,population_size - 3,2):
                population[j], population[j+1] = crossover(population[j-1],population[j],crossover_rate)

            for j in range(population_size - 3,population_size):
                population[j] = init_windows()

            print("Generation: {}".format(i))
            print("Top fitness: {}".format(top[0][0]))

    except KeyboardInterrupt:
        pass

    decoded = decode(population[top[0][1]],4,seed)
    img = Image.fromarray(decoded)
    img.show()
    img2 = Image.fromarray(target)
    img2.show()
