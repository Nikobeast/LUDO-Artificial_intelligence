import ludopy
import numpy as np
import random
from tqdm import tqdm
import json
from ludoPlayer import Player
from generic_algorithm import GenAI
from LUDO_real import *

class game:
    def __init__(self):
        self.g = ludopy.Game()
        self.there_is_a_winner = False
        self.dice = None
        self.player_i = None
        self.move_pieces = []
        self.current_position = []
        self.enemy_pieces = []
        self.player = [Player(), Player(), Player(), Player()]

    def run_game(self):

        while not self.there_is_a_winner:
            (self.dice, self.move_pieces, self.current_position, self.enemy_pieces, _, self.there_is_a_winner), self.player_i = self.g.get_observation()

            if self.player_i == 0:
                _, _, _, _, _, self.there_is_a_winner = self.g.answer_observation(self.player[0].return_action(self.dice, self.move_pieces, self.current_position, self.enemy_pieces))
            if self.player_i == 1:
                _, _, _, _, _, self.there_is_a_winner = self.g.answer_observation(self.player[1].return_action(self.dice, self.move_pieces, self.current_position, self.enemy_pieces))
            if self.player_i == 2:
                _, _, _, _, _, self.there_is_a_winner = self.g.answer_observation(self.player[2].return_action(self.dice, self.move_pieces, self.current_position, self.enemy_pieces))
            if self.player_i == 3:
                _, _, _, _, _, self.there_is_a_winner = self.g.answer_observation(self.player[3].return_action(self.dice, self.move_pieces, self.current_position, self.enemy_pieces))

        self.there_is_a_winner = False
        self.g.reset()
        return self.player_i


if __name__ == '__main__':
    # Intern fight, Child from mating, train new child, Test best chromosome vs 3 random
    # AI = game()
    # AI.player[0] = Player("finalQtable0.json", "chromosome0.npy")
    # AI.player[1] = Player("finalQtable1.json", "chromosome1.npy")
    # AI.player[2] = Player("finalQtable2.json", "chromosome2.npy")
    # AI.player[3] = Player("finalQtable3.json", "chromosome3.npy")
    # counter = 0
    # train_iterations = 5001
    # competition_iterations = 1001
    # while True:
    #     max_interations = 1001
    #     winners = []
    #     for j in tqdm(range(1, max_interations)):
    #         winner = AI.run_game()
    #         winners.append(winner)
    #         if j % 100 == 0:
    #             print("Fight amongst genes")
    #             print((np.size(np.where(np.asarray(winners) == 0))/j)*100)
    #             print((np.size(np.where(np.asarray(winners) == 1)) / j) * 100)
    #             print((np.size(np.where(np.asarray(winners) == 2)) / j) * 100)
    #             print((np.size(np.where(np.asarray(winners) == 3)) / j) * 100)
    #     #  print(np.size(np.where(np.asarray(winners) == 0)))
    #     #  print(max_interations)
    #     wins = [np.count_nonzero(np.asarray(winners) == 0), np.count_nonzero(np.asarray(winners) == 1),
    #             np.count_nonzero(np.asarray(winners) == 2), np.count_nonzero(np.asarray(winners) == 3)]
    #     winner = int(np.argmax(wins))
    #     loser = int(np.argmin(wins))
    #     gen = GenAI(np.asarray(AI.player[winner].chromosome))
    #
    #     print("best ", AI.player[winner].getchromosome())
    #
    #     # without mating
    #     # newchromosome = gen.mutate_chromosome(AI.player[winner].getchromosome())
    #
    #     # with mating - finds the second best chromosome
    #     x = np.sort(np.asarray(wins))
    #     sec = np.where(np.asarray(wins) == x[2])
    #     #  print("sec-best ", AI.player[int(sec[0])].getchromosome())
    #     newchromosome = gen.pair_mating(AI.player[winner].getchromosome(), AI.player[int(sec[0][0])].getchromosome())
    #
    #
    #     print("mutated child ", newchromosome)
    #     #  np.save("chromosome"+str(loser), newchromosome)
    #
    #     print("Train new Chromosome vs 3xRandomPlayer")
    #     LUDO_trainer = trainNewChild(newchromosome, loser, train_iterations)
    #
    #     AI.player[loser] = Player("finalQtable" + str(loser)+".json", "chromosome"+str(loser)+".npy")
    #
    #     print("Number of generations: ", counter)
    #     counter += 1
    #
    #     FinalAI = game()
    #     FinalAI.player[0] = Player("finalQtable" + str(winner) + ".json", "chromosome" + str(winner) + ".npy")
    #
    #     winners = []
    #     for j in tqdm(range(1, competition_iterations)):
    #         winner = FinalAI.run_game()
    #         winners.append(winner)
    #         if j % 100 == 0:
    #             print("Winner genes vs 3xRandomPlayer")
    #             print((np.size(np.where(np.asarray(winners) == 0)) / j) * 100)
    #             print((np.size(np.where(np.asarray(winners) == 1)) / j) * 100)
    #             print((np.size(np.where(np.asarray(winners) == 2)) / j) * 100)
    #             print((np.size(np.where(np.asarray(winners) == 3)) / j) * 100)
    #     print("The current Win rate of the best Chromosome is:")
    #     print((np.size(np.where(np.asarray(winners) == 0)) / competition_iterations) * 100)
    #     with open('BestChromsomeVs3Random.txt', 'a+') as f:
    #         f.write(str((np.size(np.where(np.asarray(winners) == 0)) / competition_iterations) * 100)+', '
    #                 + str((np.size(np.where(np.asarray(winners) == 1)) / competition_iterations) * 100)+', '
    #                 + str((np.size(np.where(np.asarray(winners) == 2)) / competition_iterations) * 100)+', '
    #                 + str((np.size(np.where(np.asarray(winners) == 3)) / competition_iterations) * 100)+'\n')
    #
    #     if (np.size(np.where(np.asarray(winners) == 0)) / competition_iterations) * 100 >= 50:
    #         print("_______________________________________________________")
    #         print("DONE MORE THAN 50 % win vs 3 random players!")
    #         print('POSSIBLE NEW BEST SCORE')
    #         print((np.size(np.where(np.asarray(winners) == 0)) / competition_iterations) * 100)
    #         print("best mutated ", newchromosome)
    #
    #         with open("WinnerQtable.json", "w") as f:
    #             k = FinalAI.player[0].qtable.keys()
    #             v = FinalAI.player[0].qtable.values()
    #             k1 = [str(i) for i in k]
    #             json.dump(json.dumps(dict(zip(*[k1, v]))), f)
    #         print('final winner chromosome', FinalAI.player[0].chromosome)
    #         np.save("WinnerChromosome", FinalAI.player[0].chromosome)
    #         # print("DONE")
    #         print("_______________________________________________________")
    #
    #     if counter > 50:
    #         print("_______________________________________________________")
    #         print("DONE 50 GENERATIONS")
    #         print((np.size(np.where(np.asarray(winners) == 0)) / competition_iterations) * 100)
    #         print("best mutated ", newchromosome)
    #
    #         with open("WinnerQtable.json", "w") as f:
    #             k = FinalAI.player[0].qtable.keys()
    #             v = FinalAI.player[0].qtable.values()
    #             k1 = [str(i) for i in k]
    #             json.dump(json.dumps(dict(zip(*[k1, v]))), f)
    #         print('final winner chromosome', FinalAI.player[0].chromosome)
    #         np.save("WinnerChromosome", FinalAI.player[0].chromosome)
    #         print("DONE")
    #         print("_______________________________________________________")
    #         break


    # ------------------------------------------------------------------------------------------------------------------
    # Each chromosome vs 3 random 1k iterations, develop child, raise child, replace lowest score chromosome*
    AI = game()
    AI.player[0] = Player("finalQtable0.json", "chromosome0.npy")
    AI.player[1] = Player("finalQtable1.json", "chromosome1.npy")
    AI.player[2] = Player("finalQtable2.json", "chromosome2.npy")
    AI.player[3] = Player("finalQtable3.json", "chromosome3.npy")
    counter = 0
    best_win_rate = 50
    train_iterations = 8001
    while True:
        max_interations = 1001
        wins = []

        for index in range(4):
            FinalAI = game()
            winners = []
            FinalAI.player[index] = Player("finalQtable" + str(index) + ".json", "chromosome" + str(index) + ".npy")
            for j in tqdm(range(1, max_interations)):
                winner = FinalAI.run_game()
                winners.append(winner)
                if j % 100 == 0:
                    print("Fight amongst genes")
                    print((np.size(np.where(np.asarray(winners) == 0))/j) * 100)
                    print((np.size(np.where(np.asarray(winners) == 1)) / j) * 100)
                    print((np.size(np.where(np.asarray(winners) == 2)) / j) * 100)
                    print((np.size(np.where(np.asarray(winners) == 3)) / j) * 100)
            wins.append((np.size(np.where(np.asarray(winners) == index)) / j) * 100)
            with open('BestChromsomeVs3Random'+str(index)+'.txt', 'a+') as f:
                f.write(str((np.size(np.where(np.asarray(winners) == 0)) / max_interations) * 100)+', '
                        + str((np.size(np.where(np.asarray(winners) == 1)) / max_interations) * 100)+', '
                        + str((np.size(np.where(np.asarray(winners) == 2)) / max_interations) * 100)+', '
                        + str((np.size(np.where(np.asarray(winners) == 3)) / max_interations) * 100)+'\n')
            with open('ChromosomeOrder'+str(index)+'.txt', 'a+') as f:
                f.write(str(FinalAI.player[index].getchromosome())+'\n')
            del FinalAI
        print(wins)

        winner = int(np.argmax(wins))
        loser = int(np.argmin(wins))
        gen = GenAI(np.asarray(AI.player[winner].chromosome))
        print('winner: ', winner)
        print('loser: ', loser)

         # without mating
        # newchromosome = gen.mutate_chromosome(AI.player[winner].getchromosome())

        #  with mating - finds the second best chromosome
        x = np.sort(np.asarray(wins))
        sec = np.where(np.asarray(wins) == x[2])
        # print('sec: ', int(sec[0][0]))
        print("best: ", AI.player[winner].getchromosome())
        print("sec-best ", AI.player[int(sec[0][0])].getchromosome())
        newchromosome = gen.pair_mating(AI.player[winner].getchromosome(), AI.player[int(sec[0][0])].getchromosome())
        print("mutated child ", newchromosome)

        if wins[winner] > best_win_rate:
            print("_______________________________________________________")
            print("DONE MORE THAN 50 % win vs 3 random players!")
            print('winner: ', wins[winner])
            print("best mutated ", newchromosome)

            with open("WinnerQtable.json", "w") as f:
                k = AI.player[winner].qtable.keys()
                v = AI.player[winner].qtable.values()
                k1 = [str(i) for i in k]
                json.dump(json.dumps(dict(zip(*[k1, v]))), f)
            print('final winner chromosome', AI.player[winner].chromosome)
            np.save("WinnerChromosome", AI.player[winner].chromosome)
            print("DONE BY WINNING")
            print("_______________________________________________________")
            best_win_rate = wins[winner]
            #  break

        print("Train new Chromosome vs 3xRandomPlayer")
        LUDO_trainer = trainNewChild(newchromosome, loser, train_iterations)

        AI.player[loser] = Player("finalQtable" + str(loser)+".json", "chromosome"+str(loser)+".npy")

        print("Number of generations: ", counter)
        counter += 1

        if counter == 50:
            print("_______________________________________________________")
            print("DONE WITH THAN 50 GENERATIONS")
            print('winner: ', wins[winner])
            print('best win rate: ', best_win_rate)
            print("best mutated ", newchromosome)

            with open("WinnerQtable50.json", "w") as f:
                k = AI.player[winner].qtable.keys()
                v = AI.player[winner].qtable.values()
                k1 = [str(i) for i in k]
                json.dump(json.dumps(dict(zip(*[k1, v]))), f)
            print('final winner chromosome', AI.player[winner].chromosome)
            np.save("WinnerChromosome50", AI.player[winner].chromosome)
            print("DONE BY GENERATIONS")
            print("_______________________________________________________")
            break

    # # Test of 1337 game Chromosome (best chromosome test) ----------------------------------------------------------
    # AI = game()
    # AI.player[0] = Player("finalQtable1337.json", "WinnerChromosome.npy")
    #
    # while True:
    #     max_interations = 10001
    #     wins = []
    #
    #     winners = []
    #     for j in tqdm(range(1, max_interations)):
    #         winner = AI.run_game()
    #         winners.append(winner)
    #         if j % 100 == 0:
    #             print("Fight amongst genes")
    #             print((np.size(np.where(np.asarray(winners) == 0))/j)*100)
    #             print((np.size(np.where(np.asarray(winners) == 1)) / j) * 100)
    #             print((np.size(np.where(np.asarray(winners) == 2)) / j) * 100)
    #             print((np.size(np.where(np.asarray(winners) == 3)) / j) * 100)
    #             with open('1337vs3Random.txt', 'a+') as f:
    #                 f.write(str((np.size(np.where(np.asarray(winners) == 0)) / j) * 100) + ', '
    #                         + str((np.size(np.where(np.asarray(winners) == 1)) / j) * 100) + ', '
    #                         + str((np.size(np.where(np.asarray(winners) == 2)) / j) * 100) + ', '
    #                         + str((np.size(np.where(np.asarray(winners) == 3)) / j) * 100) + '\n')
    #     print(np.size(np.where(np.asarray(winners) == 0)))
    #     print(max_interations)
    #
    #     print("DONE!")
    #     print("_______________________________________________________")
    #     break

    # # test of discount factor for rapport
    # AI = game()
    # AI.player[0] = Player("./TODO_Test_gamma_vs_best_vs_std/finalQtable1337_0.json", "./TODO_Test_gamma_vs_best_vs_std/chromosome1337_0.npy")
    # AI.player[1] = Player("./TODO_Test_gamma_vs_best_vs_std/finalQtable1337_05.json", "./TODO_Test_gamma_vs_best_vs_std/chromosome1337_05.npy")
    # AI.player[2] = Player("./TODO_Test_gamma_vs_best_vs_std/finalQtable1337_09.json", "./TODO_Test_gamma_vs_best_vs_std/chromosome1337_09.npy")
    # AI.player[3] = Player("./TODO_Test_gamma_vs_best_vs_std/finalQtablestandard.json", "./TODO_Test_gamma_vs_best_vs_std/chromosomestandard.npy")
    #
    # while True:
    #     max_interations = 10001
    #     wins = []
    #
    #     winners = []
    #     for j in tqdm(range(1, max_interations)):
    #         winner = AI.run_game()
    #         winners.append(winner)
    #         if j % 100 == 0:
    #             print("Fight amongst genes")
    #             print((np.size(np.where(np.asarray(winners) == 0))/j)*100)
    #             print((np.size(np.where(np.asarray(winners) == 1)) / j) * 100)
    #             print((np.size(np.where(np.asarray(winners) == 2)) / j) * 100)
    #             print((np.size(np.where(np.asarray(winners) == 3)) / j) * 100)
    #             with open('Test_of_discount_factor.txt', 'a+') as f:
    #                 f.write(str((np.size(np.where(np.asarray(winners) == 0)) / j) * 100) + ', '
    #                         + str((np.size(np.where(np.asarray(winners) == 1)) / j) * 100) + ', '
    #                         + str((np.size(np.where(np.asarray(winners) == 2)) / j) * 100) + ', '
    #                         + str((np.size(np.where(np.asarray(winners) == 3)) / j) * 100) + '\n')
    #     print(np.size(np.where(np.asarray(winners) == 0)))
    #     print(max_interations)
    #
    #     print("DONE!")
    #     print("_______________________________________________________")
    #     break





