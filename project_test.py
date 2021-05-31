from project_env import *
import numpy as np
import matplotlib.pyplot as plt
import random

### ENVIRONMENT ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n = 20 #city grid size
actions = 26 #number of actions 

City = np.zeros([n,n]) #initialize city
Q = np.zeros((n,n,actions)) #initialize Q with n x n environment and 26 actions

### CITY CREATION ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
building_heights = np.arange(0,550,50) #array of possible building heights 
weights=[15,20,15,10,10,5,5,5,5,5,5] #probabilities of each building height

random.seed(0)
for i in range(n):
    for j in range(n):
        #City[i,j] = 50*random.randint(0,10)
        height = random.choices(building_heights,weights)
        City[i,j] = height[0]

#street locations with height 0   
r1,r2,r3,r4 = 3,7,12,16 #rows
c1,c2,c3,c4 = 3,7,12,16 #columns      
City[r1,:] = 0
City[r2,:] = 0
City[r3,:] = 0
City[r4,:] = 0
City[:,c1] = 0
City[:,c2] = 0
City[:,c3] = 0
City[:,c4] = 0

fig, ax = plt.subplots()
im = ax.imshow(City)
plt.colorbar(im)
plt.ylim(-0.5,19.5)
plt.xlim(-0.5,19.5)
plt.yticks(np.arange(0,20,1))
plt.xticks(np.arange(0,20,1))

#plt.grid()

### ACTIONS ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Forward = F, Backward = B, Right = R, Left = left, Up = U, Down = D 
#actions correspond to 0 through 25, respectively 

Actions = ['F','B','R','L','U','D','FR','FL','FU','FD','BR','BL','BU','BD',
           'RU','RD','LU','LD','FRU','FRD','FLU','FLD','BRU','BRD','BLU','BLD']

### FUNCTIONS ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#gives random start location and delivery location with the elevation
def start_delivery_locations():
  #get a random row and column index
  start_row = np.random.randint(n)
  start_column = np.random.randint(n)
  delivery_row = np.random.randint(n)
  delivery_column = np.random.randint(n)
  
  start_height = City[start_row,start_column]
  delivery_height = City[delivery_row,delivery_column]

  while start_row in [r1,r2,r3,r4] or start_column in [c1,c2,c3,c4]\
      or start_height >= 400:
      start_row = np.random.randint(n)
      start_column = np.random.randint(n)
      start_height = City[start_row,start_column]
  while delivery_row in [r1,r2,r3,r4] or delivery_column in [c1,c2,c3,c4]\
      or delivery_height >= 400:
      delivery_row = np.random.randint(n)
      delivery_column = np.random.randint(n)
      delivery_height = City[delivery_row,delivery_column]

  return start_row,start_column,start_height,\
         delivery_row,delivery_column,delivery_height
         
#compares drone height to building height, works as a terminal state function
def height_reward(height,row_index,column_index):
    if height >= 400 or height <= 100 or height <= City[row_index,column_index]:
        #rewards[row_index,column_index] = -100
        #return False
        return -100
    else:
        #rewards[row_index,column_index] = -1
        #return True
        return -1

#factors in effect of wind 
def wind(height,row_index,column_index,ep):
    #right = 0, left = 1, up = 2, down = 3, no wind = 4
    direction = np.random.randint(5)
    if direction == 0: #wind coming from right
        #wind blows drone into building
        if height_reward(height,row_index,column_index-1) == -100 \
            or column_index == 0:
            return False,3 #'L'
        #adjacent building blocks wind 
        elif height < City[row_index,column_index+1]:
            return True, new_action(row_index,column_index,ep)
        #wind blows drone in direction of wind without hitting building
        else:
            return True, 3 #'L'
    if direction == 1: #wind coming from left
        if height_reward(height,row_index,column_index+1) == -100 \
            or column_index == 19:
            return False, 2 #'R'
        elif height < City[row_index,column_index-1]:
            return True, new_action(row_index,column_index,ep)
        else:
            return True, 2 #'R'
    if direction == 2: #wind coming from above/up
        if height_reward(height,row_index-1,column_index) == -100 \
            or row_index == 0:
            return False, 5 #'D'
        elif height < City[row_index+1,column_index]:
            return True, new_action(row_index,column_index,ep)
        else:
            return True, 5 #'D'
    if direction == 3: #wind coming from below/down
        if height_reward(height,row_index+1,column_index) == -100 \
            or row_index == 19:
            return False, 4 #'U'
        elif height < City[row_index-1,column_index]:
            return True, new_action(row_index,column_index,ep)
        else:
            return True, 4 #'U'
    if direction == 4: #no wind
        return True, new_action(row_index,column_index,ep)

#next action using epsilon greedy
def new_action(row_index,column_index,ep):
  if np.random.random() < ep:
    return np.argmax(Q[row_index,column_index]) #standard Q-learning
  else: 
    return np.random.randint(actions) #exploration 

#next location based on action 
def new_location(height,row_index,column_index,action):
  new_row_index = row_index
  new_column_index = column_index
  new_height = height 
  
  if Actions[action] == 'F' and new_row_index < n-1:
      new_row_index += 1
  elif Actions[action] == 'B' and new_row_index > 0:
      new_row_index -= 1
  elif Actions[action] == 'R' and new_column_index < n-1:
      new_column_index += 1
  elif Actions[action] == 'L' and new_column_index > 0:
      new_column_index -= 1
  elif Actions[action] == 'U' and new_height < 400:
      new_height += 50 
  elif Actions[action] == 'D' and new_height > 100:
      new_height -= 50
  elif Actions[action] == 'FR' and new_row_index < n-1 and new_column_index < n-1:
      new_row_index += 1
      new_column_index += 1
  elif Actions[action] == 'FL' and new_row_index < n-1 and new_column_index > 0:
      new_row_index += 1
      new_column_index -= 1
  elif Actions[action] == 'FU' and new_row_index < n-1 and new_height < 400:
      new_row_index += 1
      new_height += 50 
  elif Actions[action] == 'FD' and new_row_index < n-1 and new_height > 100:
      new_row_index += 1
      new_height -= 50
  elif Actions[action] == 'BR' and new_row_index > 0 and new_column_index < n-1:
      new_row_index -= 1
      new_column_index += 1
  elif Actions[action] == 'BL' and new_row_index > 0 and new_column_index > 0:
      new_row_index -= 1
      new_column_index -= 1
  elif Actions[action] == 'BU' and new_row_index > 0 and new_height < 400:
      new_row_index -= 1
      new_height += 50
  elif Actions[action] == 'BD' and new_row_index > 0 and new_height > 100:
      new_row_index -= 1
      new_height -= 50
  elif Actions[action] == 'RU' and new_column_index < n-1 and new_height < 400:
      new_column_index += 1
      new_height += 50
  elif Actions[action] == 'RD' and new_column_index < n-1 and new_height > 100:
      new_column_index += 1
      new_height -= 50
  elif Actions[action] == 'LU' and new_column_index > 0 and new_height < 400:
      new_column_index -= 1
      new_height += 50
  elif Actions[action] == 'LD' and new_column_index > 0 and new_height > 100:
      new_column_index -= 1
      new_height -= 50
  elif Actions[action] == 'FRU' and new_row_index < n-1 and new_column_index < n-1\
      and new_height < 400:
      new_row_index += 1
      new_column_index += 1
      new_height += 50
  elif Actions[action] == 'FRD' and new_row_index < n-1 and new_column_index < n-1\
      and new_height > 100:
      new_row_index += 1
      new_column_index += 1
      new_height -= 50
  elif Actions[action] == 'FLU' and new_row_index < n-1 and new_column_index > 0\
      and new_height < 400:
      new_row_index += 1
      new_column_index -= 1
      new_height += 50
  elif Actions[action] == 'FLD' and new_row_index < n-1 and new_column_index > 0\
      and new_height > 100:
      new_row_index += 1
      new_column_index -= 1
      new_height -= 50
  elif Actions[action] == 'BRU' and new_row_index > 0 and new_column_index < n-1\
      and new_height < 400:
      new_row_index -= 1
      new_column_index += 1
      new_height += 50
  elif Actions[action] == 'BRD' and new_row_index > 0 and new_column_index < n-1\
      and new_height > 100:
      new_row_index -= 1
      new_column_index += 1
      new_height -= 50
  elif Actions[action] == 'BLU' and new_row_index > 0 and new_column_index > 0\
      and new_height < 400:
      new_row_index -= 1
      new_column_index -= 1
      new_height += 50
  elif Actions[action] == 'BLD' and new_row_index > 0 and new_column_index > 0\
      and new_height > 100:
      new_row_index -= 1
      new_column_index -= 1
      new_height -= 50
  return new_height, new_row_index, new_column_index
  
    
#Plotting start and end points     
#getting start and delivery locations
s_r,s_c,s_h,d_r,d_c,d_h = start_delivery_locations() 

plt.scatter(s_r, s_c, s=20, c='red', marker='o')
plt.scatter(d_r, d_c, s=50, c='red', marker='x')
plt.show()

### REWARDS ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#rewards for each state
rewards = -np.ones((n,n)) #set penalty to -1 by default
rewards[d_r, d_c] = 100 #successful delivery reward

rewards = np.flip(rewards.T,0)

### Q-LEARNING ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ep = 1 #epsilon - decrease for exploration  
gamma = 0.99 #discount factor 
alpha = 0.01 #learning rate
#episodes = #number of episodes
