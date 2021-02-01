#!/bin/python3

"""
    This script runs simulations of the Monty Hall Problem
"""

import random
import matplotlib.pyplot as plt

def monty_hall_trial(n_doors, n_doors_opened, switch=True):
    
    # compute random winning door and initial guess:
    actual = random.randrange(n_doors)
    guess = random.randrange(n_doors)
     
    if switch:        
        # generate a list of other doors:
        other_doors = list(range(n_doors))
        other_doors.remove(actual)
        if actual != guess:
            other_doors.remove(guess)
        
        # switch to an unopened door:
        unopened_doors = random.sample(other_doors,len(other_doors)-n_doors_opened)
        if actual != guess:
            unopened_doors.append(actual)
        guess = random.choice(unopened_doors)
    
    return (guess == actual)

def main():
    
    N_DOORS = 3
    N_DOORS_OPENED = 1
    N_TRIALS = 10000
    
    switch_policy = { True : 0, False : 0 }
    stay_policy =   { True : 0, False : 0 }
    switch_probs = []
    stay_probs = []

    for _ in range(N_TRIALS):
        
        switch_result = monty_hall_trial(N_DOORS,N_DOORS_OPENED,switch=True)
        stay_result = monty_hall_trial(N_DOORS,N_DOORS_OPENED,switch=False)
        switch_policy[switch_result] += 1
        stay_policy[stay_result] += 1
        
        switch_probs.append(switch_policy[True]/(switch_policy[True]+switch_policy[False]))
        stay_probs.append(stay_policy[True]/(stay_policy[True]+stay_policy[False]))    

    plt.plot(switch_probs, 'r', label=f'switch (p={switch_probs[-1]}')
    plt.plot([switch_probs[-1]]*N_TRIALS, 'k--')
    plt.plot(stay_probs, 'b', label=f'no switch  (p={stay_probs[-1]})')
    plt.plot([stay_probs[-1]]*N_TRIALS, 'k--')
    plt.ylim(0.0,1.0)
    plt.xlim(0,N_TRIALS)
    plt.xlabel('Number of Trials')
    plt.ylabel('Estimated Win Probability')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
    
    
