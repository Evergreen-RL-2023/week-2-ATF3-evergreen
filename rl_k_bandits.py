'''
k-bandits simulation
you can do this in O-O, functional, or imperative style
I used imperative here
or O-O, 
there would be a Bandit class, what are the methods?
intiialize bandits
loop for number of steps
choose an action
get reward
update bandit (value and counts)
update total rewards and total counts

'''
import numpy as np
import random as rnd

k = 10  # number of bandits
q_star = np.zeros([k])
rewards = np.zeros([k])
counts = np.zeros([k])



r_total = 0  # total rewards

def init_bandits():
    for i in range(k):
        q_star[i] = rnd.gauss(0, 1)



# select action 

    # if/else for choice with epsilon?


# update reward and count



def run_bandits(n_steps):
    for i in range(n_steps):
        counts[0] += 1


if __name__ == '__main__':
    rnd.seed(42)
    init_bandits()
    x = run_bandits(1000)

    for i in range(k):
        print(q_star[i])

