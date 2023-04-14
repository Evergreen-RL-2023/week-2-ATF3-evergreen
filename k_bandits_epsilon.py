import numpy as np
import random as rnd

k = 10  # number of bandits

# initialize arrays
q_star = np.zeros([k])   
rewards = np.zeros([k])
counts = np.zeros([k])

eps = 0.1  # exploration rate
alpha = 0.1  # learning rate

r_total = 0  # total rewards

def init_bandits():
    for i in range(k):
        q_star[i] = rnd.gauss(0, 1)

def select_action():          
    if rnd.random() < eps:           # choose random for exploration
        return rnd.randint(0, k - 1)
    else:                            # greedy exploitation
        return np.argmax(rewards)

def update_reward(action, reward):   # updates running avg of awards
    counts[action] += 1   
    rewards[action] += (reward - rewards[action]) / counts[action] 

def run_bandits(n_steps):
    global r_total
    for i in range(n_steps):
        action = select_action()
        reward = rnd.gauss(q_star[action], 1)
        update_reward(action, reward)
        r_total += reward

if __name__ == '__main__':
    rnd.seed(42)
    init_bandits()
    x = run_bandits(1000)
    for i in range(k):
        print(q_star[i], rewards[i])   # should format this print.. WIP
    # print(f"{'Action':<7}{'True rewards':<14}{'Estimated rewards':<20}")   # works but still aligns oddly
    # for i in range(k):
        # print(f"{i:<7}{q_star[i]:<14.2f}{rewards[i]:<20.2f}")

    print("Total rewards obtained:", r_total)