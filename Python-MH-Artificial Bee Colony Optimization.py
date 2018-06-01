############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Artificial Bee Colony Optimization

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Artificial_Bee_Colony_Optimization, File: Python-MH-Artificial Bee Colony Optimization.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Artificial_Bee_Colony_Optimization>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import random
import math
import os

# Function: Fitness Value
def fitness_calc (function_value):
    if(function_value >= 0):
        fitness_value = 1.0/(1.0 + function_value)
    else:
        fitness_value = 1.0 + abs(function_value)
    return fitness_value

# Function: Fitness Matrix
def fitness_matrix_calc(sources):
    fitness_matrix = sources.copy(deep = True)
    for i in range(0, fitness_matrix.shape[0]):
        function_value = target_function(fitness_matrix.iloc[i,0:fitness_matrix.shape[1]-2])
        fitness_matrix.iloc[i,-2] = function_value
        fitness_matrix.iloc[i,-1] = fitness_calc(function_value)
    return fitness_matrix

# Function: Initialize Variables
def initial_sources (food_sources = 3):
    sources = pd.DataFrame(np.zeros((food_sources, len(min_values))))
    sources['Function'] = 0.0
    sources['Fitness' ] = 0.0
    for i in range(0, food_sources):
        for j in range(0, len(min_values)):
            sources.iloc[i,j] = np.random.normal(0, 1, 1)[0]
    return sources

# Function: Employed Bee
def employed_bee(fitness_matrix, min_values = [-5,-5], max_values = [5,5]):
    searching_in_sources = fitness_matrix.copy(deep = True)
    new_solution = pd.DataFrame(np.zeros((1, fitness_matrix.shape[1] - 2)))
    trial        = pd.DataFrame(np.zeros((fitness_matrix.shape[0], 1)))
    for i in range(0, searching_in_sources.shape[0]):
        phi = random.uniform(-1, 1)
        j   = np.random.randint(searching_in_sources.shape[1] - 2, size = 1)[0]
        k   = np.random.randint(searching_in_sources.shape[0], size = 1)[0]
        while i == k:
            k = np.random.randint(searching_in_sources.shape[0], size=1)[0]
        xij = searching_in_sources.iloc[i, j]
        xkj = searching_in_sources.iloc[k, j]
        vij = xij + phi*(xij - xkj)
        
        for variable in range(0, searching_in_sources.shape[1] - 2):
            new_solution.iloc[0, variable] = searching_in_sources.iloc[i, variable]
        new_solution.iloc[0, j] = vij
        if (new_solution.iloc[0, j] > max_values[j]):
            new_solution.iloc[0, j] = max_values[j]
        elif(new_solution.iloc[0, j] < min_values[j]):
            new_solution.iloc[0, j] = min_values[j]
            
        new_function_value = float(target_function(new_solution.iloc[0,0:new_solution.shape[1]]))
        
        new_fitness = fitness_calc(new_function_value)
        
        if (new_fitness > searching_in_sources.iloc[i,-1]):
            searching_in_sources.iloc[i,j]  = new_solution.iloc[0, j]
            searching_in_sources.iloc[i,-2] = new_function_value
            searching_in_sources.iloc[i,-1] = new_fitness
        else:
            trial.iloc[i,0] = trial.iloc[i,0] + 1
        
        for variable in range(0, searching_in_sources.shape[1] - 2):
            new_solution.iloc[0, variable] = 0.0
            
    return searching_in_sources, trial

# Function: Probability Matrix
def probability_matrix(searching_in_sources):
    probability_values = pd.DataFrame(0, index = searching_in_sources.index, columns = ['probability','cumulative_probability'])
    source_sum = searching_in_sources['Fitness'].sum()
    for i in range(0, probability_values.shape[0]):
        probability_values.iloc[i, 0] = searching_in_sources.iloc[i, -1]/source_sum
    probability_values.iloc[0, 1] = probability_values.iloc[0, 0]
    for i in range(1, probability_values.shape[0]):
        probability_values.iloc[i, 1] = probability_values.iloc[i, 0] + probability_values.iloc[i - 1, 1]  
    return probability_values

# Function: Select Next Source
def source_selection(probability_values):
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    source = 0
    for i in range(0, probability_values.shape[0]):
        if (random <= probability_values.iloc[i, 1]):
          source = i
          break     
    return source

def outlooker_bee(searching_in_sources, probability_values, trial, min_values = [-5,-5], max_values = [5,5]):
    improving_sources = searching_in_sources.copy(deep = True)
    new_solution = pd.DataFrame(np.zeros((1, searching_in_sources.shape[1] - 2)))
    trial_update = trial.copy(deep = True)
    for repeat in range(0, improving_sources.shape[0]):
        i = source_selection(probability_values)
        phi = random.uniform(-1, 1)
        j   = np.random.randint(improving_sources.shape[1] - 2, size=1)[0]
        k   = np.random.randint(improving_sources.shape[0], size=1)[0]
        while i == k:
            k = np.random.randint(improving_sources.shape[0], size=1)[0]
        xij = improving_sources.iloc[i, j]
        xkj = improving_sources.iloc[k, j]
        vij = xij + phi*(xij - xkj)
        
        for variable in range(0, improving_sources.shape[1] - 2):
            new_solution.iloc[0, variable] = improving_sources.iloc[i, variable]
        new_solution.iloc[0, j] = vij
        if (new_solution.iloc[0, j] > max_values[j]):
            new_solution.iloc[0, j] = max_values[j]
        elif(new_solution.iloc[0, j] < min_values[j]):
            new_solution.iloc[0, j] = min_values[j]
        new_function_value = float(target_function(new_solution.iloc[0,0:new_solution.shape[1]]))
        new_fitness = fitness_calc(new_function_value)
        
        if (new_fitness > improving_sources.iloc[i,-1]):
            improving_sources.iloc[i,j]  = new_solution.iloc[0, j]
            improving_sources.iloc[i,-2] = new_function_value
            improving_sources.iloc[i,-1] = new_fitness
            trial_update.iloc[i,0] = 0
        else:
            trial_update.iloc[i,0] = trial_update.iloc[i,0] + 1
        
        for variable in range(0, improving_sources.shape[1] - 2):
            new_solution.iloc[0, variable] = 0.0    
    return improving_sources, trial_update

def scouter_bee(improving_sources, trial_update, limit = 3):
    for i in range(0, improving_sources.shape[0]):
        if (trial_update.iloc[i,0] > limit):
            for j in range(0, improving_sources.shape[1] - 2):
                improving_sources.iloc[i,j] = np.random.normal(0, 1, 1)[0]
            function_value = target_function(improving_sources.iloc[i,0:improving_sources.shape[1]-2])
            improving_sources.iloc[i,-2] = function_value
            improving_sources.iloc[i,-1] = fitness_calc(function_value)

    return improving_sources

# ABC Function
def artificial_bee_colony_optimization(food_sources = 3, iterations = 50, min_values = [-5,-5], max_values = [5,5], employed_bees = 3, outlookers_bees = 3, limit = 3):  
    count = 0
    best_value = float("inf")
    sources = initial_sources(food_sources = food_sources)
    fitness_matrix = fitness_matrix_calc(sources)
    
    while (count <= iterations):
        print("Iteration = ", count, " f(x) = ", best_value)
       
        e_bee = employed_bee(fitness_matrix, min_values = min_values, max_values = max_values)
        for i in range(0, employed_bees - 1):
            e_bee = employed_bee(e_bee[0], min_values = min_values, max_values = max_values)
        probability_values = probability_matrix(e_bee[0])
            
        o_bee = outlooker_bee(e_bee[0], probability_values, e_bee[1], min_values = min_values, max_values = max_values)
        for i in range(0, outlookers_bees - 1):
            o_bee = outlooker_bee(o_bee[0], probability_values, o_bee[1], min_values = min_values, max_values = max_values)

        if (best_value > o_bee[0].iloc[o_bee[0]['Function'].idxmin(),-2]):
            best_solution = o_bee[0].iloc[o_bee[0]['Function'].idxmin(),:].copy(deep = True)
            best_value = o_bee[0].iloc[o_bee[0]['Function'].idxmin(),-2]
       
        sources = scouter_bee(o_bee[0], o_bee[1], limit = limit)
        fitness_matrix = fitness_matrix_calc(sources)
        
        count = count + 1   
    print(best_solution[0:len(best_solution)-1])
    return best_solution[0:len(best_solution)-1]

######################## Part 1 - Usage ####################################
 
# Function to be Minimized. Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def target_function (variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

artificial_bee_colony_optimization(food_sources = 20, iterations = 1000, min_values = [-5,-5], max_values = [5,5], employed_bees = 20, outlookers_bees = 20, limit = 40)
