#Imports
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os.path

#TODO
# - vary seed in genes
# - Allow for gene of different size?
# - non-random initalisation
# - Crossover on pixel level
# - Roulette wheel selection!
# - RGB
# - Parallelisation of decoding
# - Hyperparameter tuning
# - Remove whitespace matplotlib plot
# - Parallelisation
# - Loss per Generation Graph
# - Save state
# - Comments
# - Elitism
# - Make functions from main

#Hyper parameters
gene_size = 256
population_size = 50
mutation_rate = 2/population_size
crossover_rate = 1/population_size
generations = 24000

fractal_iterations = 3
rule_size_sqrt = 3
dimensions = 3
seed = 100
elites = 1
randoms = 1

target_image = './Images/Hand.png'
video_file = 'fname.mp4'

start_from_saved_file = False

#Initalisation of rules
#Every possible pixel value maps to a random 3x3 grid
def init_rules():
    rules = np.zeros((gene_size,rule_size_sqrt,rule_size_sqrt))
    for i in range(gene_size):
        rules[i] = np.random.randint(gene_size,size=(rule_size_sqrt,rule_size_sqrt))
    return rules

ruleset = np.array([[[11,25,25],[3,6,12],[16,15,11]],[[26,26,28],[25,27,29],[28,29,29]]
,[[28,28,29],[22,26,26],[29,29,28]]
,[[22,29,24],[29,29,24],[29,22,21]]
,[[28,29,24],[28,26,29],[29,26,17]]
,[[0,0,2],[8,2,2],[1,3,2]]
,[[23,7,8],[18,29,12],[16,22,12]]
,[[2,2,1],[2,1,2],[1,3,1]]
,[[19,13,22],[24,18,22],[1,27,29]]
,[[0,8,23], [11,7,28], [1,5,2]]
,[[22,29,26], [27,15,19], [25,21,10]]
,[[16,17,1], [1,21,10], [13,5,29]]
,[[23,29,29], [23,29,29], [18,29,29]]
,[[27,27,29], [26,22,15], [24,12,24]]
,[[14,28,29], [28,28,14], [29,23,19]]
,[[16,7,12], [7,7,24], [29,7,29]]
,[[22,8,21], [29,21,21], [29,13,10]]
,[[4,23,23], [2,24,13], [23,0,16]]
,[[5,17,1], [27,8,2], [5,25,1]]
,[[25,25,15], [19,29,29], [19,28,21]]
,[[1,5,0], [1,6,2], [3,0,2]]
,[[27,27,20], [22,22,27], [29,22,28]]
,[[1,1,1], [4,3,2], [3,2,2]]
,[[29,27,24], [29,28,24], [29,26,24]]
,[[21,25,27], [23,28,27], [19,28,29]]
,[[28,28,28], [21,13,26], [3,7,27]]
,[[5,22,20], [15,22,15], [24,14,22]]
,[[1,3,2], [2,1,7], [3,2,0]]
,[[3,15,10], [22,7,29], [5,22,7]]
,[[1,1,3], [1,1,2], [1,1,3]]])

#Decodes rule set into image
#An image is grown from a seed pixel value with the provided rule set
def decode(rules,fractal_iterations,seed):
    seed_matrix = np.array([[seed]])

    for i in range(fractal_iterations):
        size_y = seed_matrix.shape[0]
        size_x = seed_matrix.shape[1]
        new_matrix = np.zeros((size_y*rule_size_sqrt,size_x*rule_size_sqrt))

        for i in range(size_y):
            for j in range(size_x):
                seed_value = seed_matrix[i,j]
                new_matrix[i*rule_size_sqrt : i*rule_size_sqrt+rule_size_sqrt, j*rule_size_sqrt : j*rule_size_sqrt+rule_size_sqrt] = rules[int(seed_value)]
        seed_matrix = new_matrix

    return seed_matrix

#The the sum of the pixel-wise absolute differences between the target image and the image created using the rule set is used as fitness measure. A sum of 0 means that the target image could be recreated from the rule set without loss. The largest possible difference is the gene size x height x width of the image.
def fitness(target_image,genetic_image):
    difference = np.subtract(target_image,genetic_image)
    square = np.square(difference)
    sum = np.sum(square)

    return sum

#Each pixel is mutated with a mutation_rate/2 chance.
#The range is clipped to a minimum of 0 and a maximum of gene size.
def mutate(rules,mutation_rate,gene_size,rule_size_sqrt):
    mutated_rules = np.zeros((gene_size,rule_size_sqrt,rule_size_sqrt))
    for i in range(gene_size):
        random_matrix = np.random.choice([-1,0,1],(rule_size_sqrt,rule_size_sqrt),[mutation_rate/2,1-mutation_rate,mutation_rate/2])
        mutated_rules[i] = np.clip(rules[i] + random_matrix,a_min=0,a_max=255)
    return mutated_rules

#Each rule in the rule sets is crossover at the crossover rate.
def crossover(rules1,rules2,crossover_rate):
    crossover_rules1 = np.zeros((gene_size,rule_size_sqrt,rule_size_sqrt))
    crossover_rules2 = np.zeros((gene_size,rule_size_sqrt,rule_size_sqrt))
    for i in range(gene_size):
        randi = np.random.choice([0,1],p=[1-crossover_rate,crossover_rate])
        if(randi):
            crossover_rules1[i] = rules2[i]
            crossover_rules2[i] = rules1[i]
        else:
            crossover_rules1[i] = rules1[i]
            crossover_rules2[i] = rules2[i]

    return crossover_rules1, crossover_rules2

if __name__ == '__main__':
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Fractal-Compression')
    writer = FFMpegWriter(fps=24, metadata=metadata)

    fig = plt.figure()
    plt.axis('off')

    img = Image.open(target_image).convert("L")
    target = np.asarray(img)

    if(os.path.isfile('Hand.npy') and start_from_saved_file):
        population = np.load('Hand.npy')
    else:
        population = np.zeros((population_size,gene_size,rule_size_sqrt,rule_size_sqrt))

        for i in range(population_size):
            population[i] = init_rules()

    writepops = []

    try:
        for i in range(generations):
            top = []
            total_fitness = 0
            for j in range(population_size):
                decoded = decode(population[j],fractal_iterations,seed)
                decoded_fitness = fitness(decoded,target)
                total_fitness += decoded_fitness
                top.append((decoded_fitness,j))
            top_sorted = sorted(top, key=lambda tup: tup[0])

            odds = []

            for j in range(population_size):
                odds.append(top[j][0] / total_fitness)

            old_population = population

            for j in range(elites):
                population[j] = population[top_sorted[j][1]]

            for j in range(elites,population_size):
                population[j] = mutate(population[j],mutation_rate,gene_size,rule_size_sqrt)

            for j in range(elites,population_size):
               random_pop = np.random.choice(population_size,1,odds)
               population[j] = old_population[random_pop]

            for j in range(elites,population_size - randoms,2):
                random_indices = np.random.choice(population_size,2,odds)
                population[j], population[j+1] = crossover(old_population[random_indices[0]],old_population[random_indices[1]],crossover_rate)

            for j in range(population_size - randoms,population_size):
                population[j] = init_rules()

            new_fitness = top_sorted[0][0]

            print("Generation: {}".format(i))
            print("Top fitness: {}".format(new_fitness))

            if(i==0 or new_fitness<old_fitness):
                old_fitness = new_fitness
                writepops.append(decode(population[0],fractal_iterations,seed))


    except KeyboardInterrupt:
        pass

    with writer.saving(fig, video_file ,100):
        total_frames = len(writepops)
        for idx, wrtpop in enumerate(writepops):
            print("{}/{}".format(idx,total_frames))
            plt.imshow(wrtpop,cmap='Greys_r',interpolation='nearest')
            writer.grab_frame()
    np.save('Hand',population)
