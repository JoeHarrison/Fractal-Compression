#Imports
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os.path
import os


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
mutation_rate = 0.1/5
crossover_rate = 0.1/5
generations = 24000
elites = 2
randoms = 5

fractal_iterations = 3
rule_size_sqrt = 3
dimensions = 3
seed = 100

target_image = './Images/Hand.png'
video_file = 'hand.mp4'

start_from_saved_file = True
save_ruleset = True

#Initalisation of rules
#Every possible pixel value maps to a random 3x3 grid
def init_rules():
    rules = np.zeros((gene_size,dimensions,rule_size_sqrt,rule_size_sqrt))
    for i in range(gene_size):
        rules[i] = np.random.randint(gene_size,size=(dimensions,rule_size_sqrt,rule_size_sqrt))
    return rules

#Decodes rule set into image
#An image is grown from a seed pixel value with the provided rule set
def decode(rules,fractal_iterations,seed):
    #Keep using numpy if possible.
    seed_matrix = [np.array([[seed]]),np.array([[seed]]),np.array([[seed]])]

    for i in range(dimensions):
        for j in range(fractal_iterations):

            size_y = seed_matrix[i].shape[0]
            size_x = seed_matrix[i].shape[1]
            new_matrix = np.zeros((size_y*rule_size_sqrt,size_x*rule_size_sqrt))

            for y in range(size_y):
                for x in range(size_x):
                    seed_value = seed_matrix[i][y,x]
                    new_matrix[y*rule_size_sqrt : y*rule_size_sqrt+rule_size_sqrt, x*rule_size_sqrt : x*rule_size_sqrt+rule_size_sqrt] = rules[int(seed_value),i]

            seed_matrix[i] = new_matrix
            #place here?
    #Image needs to be reshaped to (n,m,3) instead of (3,n,m)
    return np.moveaxis(seed_matrix,0,-1)
    #return np.array(seed_matrix).reshape((np.array(seed_matrix).shape[1], np.array(seed_matrix).shape[2],3))

#The the sum of the pixel-wise absolute differences between the target image and the image created using the rule set is used as fitness measure. A sum of 0 means that the target image could be recreated from the rule set without loss. The largest possible difference is the gene size x height x width of the image.
def fitness(target_image,genetic_image):
    penalty = 0
    for i in range(dimensions):
        difference = np.subtract(target_image[:,:,i],genetic_image[:,:,i])
        square = np.square(difference)
        penalty += np.sum(square)
    return penalty

#Each pixel is mutated with a mutation_rate/2 chance.
#The range is clipped to a minimum of 0 and a maximum of gene size.
def mutate(rules,mutation_rate,gene_size,rule_size_sqrt):
    mutated_rules = np.zeros((gene_size,dimensions,rule_size_sqrt,rule_size_sqrt))
    for i in range(gene_size):
        random_matrix = np.random.choice([-1,0,1],(dimensions,rule_size_sqrt,rule_size_sqrt),[mutation_rate/2,1-mutation_rate,mutation_rate/2])
        mutated_rules[i] = np.clip(rules[i] + random_matrix,a_min=0,a_max=255)
    return mutated_rules

#Each pixel is mutated with a random value between 0 and 255
def mutate_v2(rules,mutation_rate,gene_size,rule_size_sqrt):
    mutated_rules = np.zeros((gene_size,dimensions,rule_size_sqrt,rule_size_sqrt))
    for i in range(gene_size):
        mask = np.random.choice([0,1],(dimensions,rule_size_sqrt,rule_size_sqrt),[mutation_rate,1-mutation_rate])
        ones = np.ones((dimensions,rule_size_sqrt,rule_size_sqrt))
        inverted_mask = np.subtract(ones,mask)
        random_mask = np.ma.masked_array(np.random.randint(0,256,size=(dimensions,rule_size_sqrt,rule_size_sqrt)),mask)
        mutated_rules[i] = np.ma.masked_array(rules[i],inverted_mask) + random_mask
    return mutated_rules

def crossover_v2(rules1,rules2,crossover_rate):
    crossover_rules1 = np.zeros((gene_size,dimensions,rule_size_sqrt,rule_size_sqrt))
    crossover_rules2 = np.zeros((gene_size,dimensions,rule_size_sqrt,rule_size_sqrt))
    for i in range(gene_size):
        for j in range(dimensions):
            mask = np.random.choice([0,1],size=(rule_size_sqrt,rule_size_sqrt),p=[1-crossover_rate,crossover_rate])
            ones = np.ones((rule_size_sqrt,rule_size_sqrt))
            inverted_mask = np.subtract(ones,mask)
            crossover_rules1[i,j] = np.ma.masked_array(rules1[i,j],mask) + np.ma.masked_array(rules2[i,j],inverted_mask)
            crossover_rules2[i,j] = np.ma.masked_array(rules2[i,j],mask) + np.ma.masked_array(rules1[i,j],inverted_mask)

    return crossover_rules1, crossover_rules2

#Each rule in the rule sets is crossover at the crossover rate.
def crossover(rules1,rules2,crossover_rate):
    crossover_rules1 = np.zeros((gene_size,dimensions,rule_size_sqrt,rule_size_sqrt))
    crossover_rules2 = np.zeros((gene_size,dimensions,rule_size_sqrt,rule_size_sqrt))
    for i in range(gene_size):
        for j in range(dimensions):
            randi = np.random.choice([0,1],p=[1-crossover_rate,crossover_rate])
            if(randi):
                crossover_rules1[i,j] = rules2[i,j]
                crossover_rules2[i,j] = rules1[i,j]
            else:
                crossover_rules1[i,j] = rules1[i,j]
                crossover_rules2[i,j] = rules2[i,j]

    return crossover_rules1, crossover_rules2

def read_rules(file_name):
    print('Reading: {}'.format(file_name))
    with open(file_name, 'r') as f:
        seed = int(f.readline())
        gene_size = int(f.readline())
        dimensions = int(f.readline())

        rules = np.zeros((gene_size,dimensions,3,3))
        for i in range(dimensions):
            for j in range(gene_size):
                line = f.readline().split(" ")
                idx = 0
                for k in range(3):
                    for l in range(3):
                        rules[j,i,k,l] = float(line[idx])
                        idx +=1
    return rules

def write_rules(file_name,rules,seed,gene_size,dimensions):
    print('Writing: {}'.format(file_name))
    os.remove(file_name)
    with open(file_name, 'a') as f:
        f.write(str(seed) + '\n')
        f.write(str(gene_size) + '\n')
        f.write(str(dimensions) + '\n')
        for rule in rules:
            for i in range(rules.shape[1]):
                for j in range(rules.shape[0]):
                    for k in range(rules.shape[2]):
                        for l in range(rules.shape[3]):
                            f.write(str(int(rules[j,i,k,l])) + ' ')
                    f.write('\n')

if __name__ == '__main__':
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Fractal-Compression')
    writer = FFMpegWriter(fps=24, metadata=metadata)

    fig = plt.figure()
    plt.axis('off')

    img = Image.open(target_image)
    target = np.asarray(img)[:,:,:dimensions]

    if(os.path.isfile('hand.npy') and start_from_saved_file):
        population = np.load('hand.npy')
    else:
        population = np.zeros((population_size,gene_size,dimensions,rule_size_sqrt,rule_size_sqrt))

        for i in range(population_size):
            population[i] = init_rules()

    writepops = []

    try:
        for i in range(generations):
            top = []
            total_fitness = 0
            for j in range(population_size):
                decoded = decode(population[j],fractal_iterations,seed)
                decoded_fitness = fitness(target,decoded)
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
            print("Top fitness: {},  {}".format(new_fitness,np.sqrt(new_fitness)))

            if(i==0 or new_fitness<old_fitness):
                old_fitness = new_fitness
                writepops.append(decode(population[0],fractal_iterations,seed))

    except KeyboardInterrupt:
        pass

    with writer.saving(fig, video_file ,100):
        total_frames = len(writepops)
        for idx, wrtpop in enumerate(writepops):
            print("{}/{}".format(idx,total_frames))

            plt.imshow(wrtpop.astype('uint8'),interpolation='nearest')
            writer.grab_frame()

    if(save_ruleset):
        np.save('hand',population)
