import numpy as np
import random
from tqdm import tqdm
import json



class Player:
    def __init__(self, QtableName = "", chromosomeName = ""):
        self.enable_randomplayer = False
        if QtableName == "" or chromosomeName == "":
            self.enable_randomplayer = True
        else:
            self.qtable = {}  # init a dictionary
            with open(QtableName, "r") as f:
                data = json.load(f)
                dic = json.loads(data)
                k = dic.keys()
                v = dic.values()
                k1 = [eval(i) for i in k]
                self.qtable = dict(zip(*[k1, v]))
            self.chromosome = np.load(chromosomeName)  # Det burde virke her nu...
            self.discount_factor = self.chromosome[0] # 0.9  # gamma
            self.learning_rate = self.chromosome[1] # 0.5  # alpha
            self.epsilon = 0  # 100% greedy
            self.reward = 0  # r
            self.HOME_AREAL_INDEXS = np.array([53, 54, 55, 56, 57, 58])
            self.GLOBUS_INDEXS = np.array([9, 22, 35, 48])
            self.STAR_INDEXS = np.array([5, 12, 18, 25, 31, 38, 44, 51])
            self.GOAL_INDEX = 59
            self.ENEMY_1_GLOB_INDX = 14
            self.ENEMY_2_GLOB_INDX = 27
            self.ENEMY_3_GLOB_INDX = 40
            self.next_position = []
            self.next_qvalue = 0
            self.there_is_a_winner = False
            self.dice = None
            self.player_i = None
            self.current_position = []
            self.enemy_pieces = []
            self.current_action = 0
            self.current_state = []
        self.move_pieces = []

    def return_action(self, dice, move_pieces, current_position, enemy_pieces):
        self.move_pieces = move_pieces
        if self.enable_randomplayer:
            if len(self.move_pieces):
                piece_to_move = self.move_pieces[np.random.randint(0, len(self.move_pieces))]
            else:
                piece_to_move = -1
            return piece_to_move
        else:
            self.dice = dice
            self.current_position = current_position
            self.enemy_pieces = enemy_pieces
            return self.find_action()

    def find_action(self):
        if len(self.move_pieces) == 0:  # There is no valid action
            return -1

        self.enemy_pieces = np.asarray(self.enemy_pieces).flatten()
        self.update_enemy_pos()

        self.cal_current_state()
        if self.qtable.get(tuple(self.current_state)) is None:
            #print("new state reached, more training is required!")
            self.qtable[tuple(self.current_state)] = self.random_vector()
        self.perform_action()  # Used current action, 0, 1, 2, 3 => gå e.g. 0-8 i current state og find reward
        return self.current_action

    # [move_to_safe_zone, safe_zone, near_enemy, release, Globus, hit_home, enemy_hit_home, reach_goal, star]
    # def return_piece_state(self):
    #     bit_vector = []
    #     if self.current_action == 0:
    #         bit_vector = self.current_state[27:36]
    #     elif self.current_action == 1:
    #         bit_vector = self.current_state[18:27]
    #     elif self.current_action == 2:
    #         bit_vector = self.current_state[9:18]
    #     elif self.current_action == 3:
    #         bit_vector = self.current_state[0:9]
    #
    #     return bit_vector

    def perform_action(self):
        if len(self.move_pieces) > 1:
            if random.random() > self.epsilon:  # explore vs exploitation
                self.current_action = self.max_action()
            else:
                index = np.random.randint(0, len(self.move_pieces))
                self.current_action = self.move_pieces[index]
        elif len(self.move_pieces) == 1:
            self.current_action = self.move_pieces[0]

    def max_action(self):
        best_index = 0
        max_val = -100000000000
        key = tuple(self.current_state)
        for i in range(len(self.qtable[key])):
            if self.qtable[key][i] > max_val and i in self.move_pieces:
                max_val = self.qtable[key][i]
                best_index = i
        return best_index

    # def get_reward(self, bit_vector):
    #     # [move_to_safe_zone, safe_zone, near_enemy, release, Globus, hit_home, enemy_hit_home, reach_goal, star]
    #     # print(bit_vector)
    #     reward = 0
    #     if bit_vector[8] == 1:  # Hit a star 9
    #         # return 0.2
    #         reward += self.chromosome[2]  # 0.2
    #     if bit_vector[7] == 1:  # Brick can hit goal 8
    #         # return 1.0
    #         reward += self.chromosome[3]  # 1.0
    #     if bit_vector[6] == 1:  # Brick can hit someone home 5
    #         # return 0.5
    #         reward += self.chromosome[4]  # 0.5
    #     if bit_vector[5] == 1:  # Brick is hit home 4
    #         # return -1.0
    #         reward -= self.chromosome[5]  # 1.0
    #     if bit_vector[4] == 1:  # Hit a Globus 10
    #         # return 0.2
    #         reward += self.chromosome[6]  # 0.2
    #     if bit_vector[3] == 1:  # Get out of home position
    #         # return 0.3
    #         reward += self.chromosome[7]  # 0.3
    #     if bit_vector[2] == 1:  # Is near an enemy 6
    #         # return -0.5
    #         reward -= self.chromosome[8]  # 0.5
    #     if bit_vector[1] == 1:  # Is in safe zone 7
    #         # return 0.4
    #         reward += self.chromosome[9]  # 0.4
    #     if bit_vector[0] == 1:  # Can move to the safe zone
    #         # return 0.05
    #         reward += self.chromosome[10]  # 0.05
    #     return reward

    def can_reach_goal(self, possible_position):
        if possible_position == self.GOAL_INDEX:  # Get to goal
            return 1
        else:
            return 0

    def can_hit_enemy_home(self, possible_position):
        if possible_position in self.enemy_pieces:  # Able to hit someone home -
            # enemy_index = self.enemy_pieces.index(possible_position)
            if possible_position not in self.GLOBUS_INDEXS and possible_position != self.ENEMY_1_GLOB_INDX and \
                    possible_position != self.ENEMY_2_GLOB_INDX and possible_position != self.ENEMY_3_GLOB_INDX \
                    and np.count_nonzero(self.enemy_pieces == possible_position) == 1:
                return 1
        return 0

    def hit_yourself_home(self, possible_position):
        if possible_position in self.enemy_pieces:  # Hit yourself home
            if possible_position in self.GLOBUS_INDEXS or possible_position == self.ENEMY_1_GLOB_INDX or \
                    possible_position == self.ENEMY_2_GLOB_INDX or possible_position == self.ENEMY_3_GLOB_INDX \
                    or np.count_nonzero(self.enemy_pieces == possible_position) > 1:
                return 1
        return 0

    def can_hit_star(self, possible_position):
        if possible_position in self.STAR_INDEXS:
            index = np.where(self.STAR_INDEXS == possible_position)
            index = int(*index)
            if index + 1 > len(self.STAR_INDEXS) - 1:
                new_possible_position = self.GOAL_INDEX
            else:
                new_possible_position = self.STAR_INDEXS[index + 1]

            return 1, new_possible_position
        return 0, possible_position

    def can_hit_globus(self, possible_position):
        if possible_position in self.GLOBUS_INDEXS:
            return 1
        else:
            return 0

    def release_piece(self, possible_position, die):
        if possible_position == 6 and die == 6:  # Brick at home and dice = 6 -> free Brick (10hi f9z)
            return 1
        else:
            return 0

    def is_near_enemy(self, possible_position):
        holder_array = np.where(abs(self.enemy_pieces) < possible_position, self.enemy_pieces, 0)
        if np.count_nonzero(holder_array) != 0:
            if np.count_nonzero(np.where(holder_array + 6 >= possible_position, 1, 0)) != 0:
                return 1
        return 0

    def in_safe_zone(self, possible_position):
        if possible_position in self.HOME_AREAL_INDEXS or possible_position > 59:  # Might need to take max position (59) into account --------------------
            return 1
        else:
            return 0

    def can_get_to_safe_zone(self, possible_position):
        if (possible_position + 6) in self.HOME_AREAL_INDEXS:
            return 1
        else:
            return 0

    # def cal_next_state(self):
    #     self.next_qvalue = -10000000000
    #     for dice in range(1, 7):
    #         next_state = np.array(np.zeros(36, dtype=int))
    #         index = 35
    #         for brick in range(0, 4):
    #             possible_position = self.next_position[brick] + dice
    #
    #             next_state[index], possible_position = self.can_hit_star(possible_position)
    #             index -= 1
    #             next_state[index] = self.can_reach_goal(possible_position)
    #             index -= 1
    #             next_state[index] = self.can_hit_enemy_home(possible_position)
    #             index -= 1
    #             next_state[index] = self.hit_yourself_home(possible_position)
    #             index -= 1
    #             next_state[index] = self.can_hit_globus(possible_position)
    #             index -= 1
    #             next_state[index] = self.release_piece(possible_position, dice)
    #             index -= 1
    #             next_state[index] = self.is_near_enemy(possible_position)
    #             index -= 1
    #             next_state[index] = self.in_safe_zone(possible_position)
    #             index -= 1
    #             next_state[index] = self.can_get_to_safe_zone(possible_position)
    #             index -= 1
    #
    #         if self.next_qvalue < self.get_max_val_from_state(next_state):
    #             self.next_qvalue = self.get_max_val_from_state(next_state)

    def cal_current_state(self):
        index = 35
        self.current_state = np.array(np.zeros(36, dtype=int))
        for brick in range(0, 4):
            possible_position = self.current_position[brick] + self.dice
            self.current_state[index], possible_position = self.can_hit_star(possible_position)
            index -= 1
            self.current_state[index] = self.can_reach_goal(possible_position)
            index -= 1
            self.current_state[index] = self.can_hit_enemy_home(possible_position)
            index -= 1
            self.current_state[index] = self.hit_yourself_home(possible_position)
            index -= 1
            self.current_state[index] = self.can_hit_globus(possible_position)
            index -= 1
            self.current_state[index] = self.release_piece(possible_position, self.dice)
            index -= 1
            self.current_state[index] = self.is_near_enemy(possible_position)
            index -= 1
            self.current_state[index] = self.in_safe_zone(possible_position)
            index -= 1
            self.current_state[index] = self.can_get_to_safe_zone(possible_position)
            index -= 1

        # print('current state: ', self.current_state)

    def get_max_val_from_state(self, state):
        max_val = -100000000000
        key = tuple(state)
        if self.qtable.get(key) is None:
            self.qtable[key] = self.random_vector()
        for i in range(len(self.qtable[key])):
            if self.qtable[key][i] > max_val:
                max_val = self.qtable[key][i]
        return max_val

    @staticmethod
    def random_vector():
        return [random.random() / 10000, random.random() / 10000, random.random() / 10000, random.random() / 10000]

    # def update_q_table(self):
    #     possible_moves = self.g.current_move_pieces
    #     key = tuple(self.current_state)
    #     if len(possible_moves) > 0:
    #         delta_q_value = self.learning_rate * (
    #                     self.reward + self.discount_factor * self.next_qvalue - self.qtable[key][self.current_action])
    #     else:
    #         delta_q_value = 0
    #     self.qtable[key][self.current_action] += delta_q_value

    def update_enemy_pos(self):
        self.enemy_pieces = np.where(self.enemy_pieces > 53, -100, self.enemy_pieces)
        self.enemy_pieces[0:4] = np.where(self.enemy_pieces[0:4] > 0, self.enemy_pieces[0:4] + 13, -100)
        self.enemy_pieces[4:8] = np.where(self.enemy_pieces[4:8] > 0, self.enemy_pieces[4:8] + 26, -100)
        self.enemy_pieces[8:12] = np.where(self.enemy_pieces[8:12] > 0, self.enemy_pieces[8:12] + 39, -100)
        self.enemy_pieces = np.where(self.enemy_pieces > 53, self.enemy_pieces - 52, self.enemy_pieces)

    def getchromosome(self):
        return self.chromosome

