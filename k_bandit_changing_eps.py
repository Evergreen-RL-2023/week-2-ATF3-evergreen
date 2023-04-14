import numpy as np
import random as rnd

k = 10  # number of bandits

# initialize arrays
q_star = np.zeros([k])   
rewards = np.zeros([k])
counts = np.zeros([k])

r_total = 0  # total rewards


def init_bandits():
    for i in range(k):
        q_star[i] = rnd.gauss(0, 1)


def select_action(eps):          
    if rnd.random() < eps:           # choose random for exploration
        return rnd.randint(0, k - 1)
    else:                            # greedy exploitation
        return np.argmax(rewards)


def update_reward(action, reward):   # updates running avg of awards
    counts[action] += 1   
    rewards[action] += (reward - rewards[action]) / counts[action] 


def run_bandits(n_steps, eps):
    r_total = 0
    for i in range(n_steps):
        action = select_action(eps)
        reward = rnd.gauss(q_star[action], 1)
        update_reward(action, reward)
        r_total += reward
    return r_total / n_steps, r_total

if __name__ == '__main__':
    rnd.seed(42)
    init_bandits()
    print(f"{'Eps':<7}{'Average rewards':<20}{'Total rewards':<20}") 
    for eps in np.arange(0.0, 1.0, 0.05):
        avg_reward, total_reward = run_bandits(1000, eps)
        print(f"{eps:<7.2f}{avg_reward:<20.2f}{total_reward:<20.2f}")