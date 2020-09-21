#!/usr/bin/env python
# coding: utf-8

# # Simulated Annealing for Combinatorial Problems

# Mohammed Agha

# June 2020

# Quadratic Assignment Problem - Code adopted from a course on Udemy

# In[4]:


# Importing relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:



# Defining dataframes for distance and flow
Distance = pd.DataFrame([[0,1,2,3,1,2,3,4],[1,0,1,2,2,1,2,3],
                         [2,1,0,1,3,2,1,2],[3,2,1,0,4,3,2,1],
                         [1,2,3,4,0,1,2,3],[2,1,2,3,1,0,1,2],
                         [3,2,1,2,2,1,0,1],[4,3,2,1,3,2,1,0]], 
                         columns = ["A", "B", "C", "D", "E", "F", "G", "H"],
                         index = ["A", "B", "C", "D", "E", "F", "G", "H"])

Flow = pd.DataFrame([[0,5,2,4,1,0,0,6],[5,0,3,0,2,2,2,0],
                     [2,3,0,0,0,0,0,5],[4,0,0,0,5,2,2,10],
                     [1,2,0,5,0,10,0,0],[0,2,0,2,10,0,5,1],
                     [0,2,0,2,0,5,0,10],[6,0,5,10,0,1,10,0]],
                     columns = ["A", "B", "C", "D", "E", "F", "G", "H"],
                     index = ["A", "B", "C", "D", "E", "F", "G", "H"])

# Initial Solution
X0 = ["B","D","A","E","C","F","G","H"]


# Hyperparameters of the Simulated Annealing algorithm
T0 = 1500
M = 250 
N = 20
alpha = 0.9

# For plotting purposes
Temp = []
OF_Value = []

# Algorithm Body
# External loop
for i in range(M):
    # Internal Loop
    for j in range(N):
        rand_1 = np.random.randint(0, len(X0))
        rand_2 = np.random.randint(0, len(X0))
        
        # We need two dif. indecies since if the two rand no. are equal we are swapping the same department by itself
        while rand_1 == rand_2:
            rand_2 = np.random.randint(0, len(X0))
        
        # define an empty temporary array to contain the candidate positions
        xt = []
        
        # candidate positions to swap
        A1 = X0[rand_1]
        A2 = X0[rand_2]
        
        # Start swapping
        w = 0
        for k in X0:
            if X0[w] == A1:
                xt = np.append(xt, A2)
            elif X0[w] == A2:
                xt = np.append(xt, A1)
            else:
                xt = np.append(xt, X0[w])
            w = w + 1
        
        # Computing the OF of the current best solution (Incumbent)
        current_Dist_DF   = Distance.reindex(columns = X0, index = X0)
        current_Dist_arr  = np.array(current_Dist_DF)
        current_OF        = pd.DataFrame(current_Dist_arr * Flow)
        current_OF_arr    = np.array(current_OF)
        current_OF_value  = sum(sum(current_OF_arr))
        
        # Computing the OF of the challenging solution (Candidate)
        candidate_Dist_DF = Distance.reindex(columns = xt, index = xt)
        candidate_Dist_arr= np.array(candidate_Dist_DF)
        candidate_OF      = pd.DataFrame(candidate_Dist_arr * Flow)
        candidate_OF_arr  = np.array(candidate_OF)
        candidate_OF_value = sum(sum(candidate_OF_arr))
        
        # decide whether to make the move
        rand_3 = np.random.rand()
        
        Formula = 1/(np.exp(candidate_OF_value - current_OF_value)/T0)
        
        if candidate_OF_value <= current_OF_value:
            X0 = xt
        elif rand_3 <= Formula:
            X0 = xt
        else:
            X0 = X0
            
    Temp.append(T0)
    OF_Value.append(current_OF_value)
    
    # Decrease temperature
    T0 = alpha * T0

print()
print("The final solution is:", X0)
print("The minimum OF value is", current_OF_value)

# Plotting
plt.plot(Temp, OF_Value)
plt.title("Cost vs. Temp.", fontsize = 20, fontweight = 'bold')
plt.xlabel("Temp.", fontsize = 18, fontweight = 'bold')
plt.ylabel("Cost", fontsize = 18, fontweight = 'bold')
plt.xlim(1500, 0)

plt.xticks(np.arange(min(Temp), max(Temp), 100), fontweight = 'bold')
plt.yticks(fontweight = 'bold')
plt.show()


# In[ ]:




