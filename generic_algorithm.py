import ludopy
import numpy as np
import random
from tqdm import tqdm


class GenAI:
    def __init__(self, chromosome):
        self.population1 = chromosome
        self.NumberGenMutation = 4
        self.randomindex = np.array([])
        self.population2 = np.array([])
        self.population3 = np.array([])
        self.population4 = np.array([])
        #self.first = first

    def pair_mating(self, parent1, parent2):
        random_idx = random.randint(1, parent1.size-1)
        print("random idx", random_idx)
        child = np.array([parent1[0:random_idx]]).flatten()
        child = np.append(child, parent2[random_idx:parent2.size])
        #  print("init child: ", child)

        mutated_child = self.mutate_chromosome(child, "mutated child")
        return mutated_child

    def return_val(self, first):
        if first:
            return self.genetic_algorithm_init()
        else:
            return self.genetic_algorithm()

    def genetic_algorithm_init(self):
        chromosome_custom = self.population1 #np.array([0.27, 0.21, 0.2, 1.0, 0.5, -1.0, 0.2, 0.3, -0.5, 0.4, 0.05])  # gamma, alpha, rewards
        self.population2 = self.mutate_chromosome(chromosome_custom)
        self.population3 = self.mutate_chromosome(chromosome_custom)
        self.population4 = self.mutate_chromosome(chromosome_custom)
        return self.population1, self.population2, self.population3, self.population4

    # def matingfunc(self):
    #     #  test of mating function
    #     chromosome_custom2 = np.array([0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5])
    #     self.pair_mating(chromosome_custom, chromosome_custom2)

    def genetic_algorithm(self):
        chromosome_custom = self.population1  # np.array([0.27, 0.21, 0.2, 1.0, 0.5, -1.0, 0.2, 0.3, -0.5, 0.4, 0.05])  # gamma, alpha, rewards
        self.population2 = self.mutate_chromosome(chromosome_custom)
        return self.population2

    def mutate_chromosome(self, chromosome, name='mutated chromosomes'):
        self.randomizator_idx(len(chromosome), self.NumberGenMutation)
        mutated = chromosome.copy()
        # print("chromosome: ", chromosome)
        while True:
            for i in self.randomindex:
                mutated[int(i)] = self.random_value(i, mutated[int(i)])  #  5 & 8 (negative)
            # print(name, mutated)
            if not np.array_equal(np.asarray(mutated), np.asarray(chromosome)):
                break
            else:
                print("chromosome is alike")
        return mutated

    def random_value(self, index, value):

        randPos = random.uniform(0.05, 0.2)
        randNeg = random.uniform(-0.2, -0.05)
        randList = [[randPos], [randNeg]]
        rand = randList[random.randint(0, 1)]

        if index == 5 or index == 8:
            if value + rand < -1:
                return -1
            elif value + rand > 0:
                return 0
            else:
                return value + rand
        else:
            if value + rand > 1:
                return 1
            elif value + rand < 0:
                return 0
            else:
                return value + rand

    def randomizator_idx(self, length, mutationKvotient):
        holder = [] #  np.append(self.randomindex, random.randint(0, 9))
        max_index = self.NumberGenMutation-1 #  random.randint(0, mutationKvotient-1)
        # print("% mutation", max_index+1)
        while True:
            rand_val = random.randint(0, length-1)
            if rand_val not in np.asarray(holder):
                holder.append(rand_val)
            if len(holder) > max_index:
                self.randomindex = np.asarray(holder)
                #  print(self.randomindex)
                break


#if __name__ == '__main__':
#    gen = GenAI(np.array([0.27, 0.21, 0.2, 1.0, 0.5, -1.0, 0.2, 0.3, -0.5, 0.4, 0.05]))
#    a = gen.return_val(False)
#    print(a)
    #print (b)
    #print(c)
    #print(d)
