import matplotlib.pyplot as plt
import numpy as np
import colorsys

def putlabels(axs, pop_fitness_col,distance):
    i = 1
    for x,y in zip(pop_fitness_col,distance):
        print(x,y,i)
        axs.text(x, y, i,
            fontsize='medium', verticalalignment='top', fontfamily='serif')
        i += 1
def generate_multilevel_diagram(population_fitness):


    max_components = []
    for i in range(len(population_fitness[0])):
        max_components.append(max(population_fitness, key= lambda x:x[i])[i])

    min_components = []
    for i in range(len(population_fitness[0])):
        min_components.append(min(population_fitness, key= lambda x:x[i])[i])

    max_components = np.asarray(max_components)
    min_components = np.asarray(min_components)

    population_fitness = np.asarray(population_fitness)

    # normalization
    normalized_fitness = np.empty(shape=(len(population_fitness),len(population_fitness[0])))

    for index,element in enumerate(population_fitness):
        normalized_fitness[index] = ((element-min_components)/(max_components-min_components))

    ideal_distance = []

    for element in normalized_fitness:
        ideal_distance.append(np.linalg.norm(element))

    fig, axs = plt.subplots(2, 2)

    N = 20
    colors = list(np.random.choice(range(256), size=N))
    colors = np.asarray(colors)
    axs[0, 0].scatter(population_fitness[:,0], ideal_distance,c=colors)
    axs[0, 0].set_title('Mon Dmg')
    axs[0, 1].scatter(population_fitness[:,1], ideal_distance,c=colors)
    axs[0, 1].set_title('Opp Dmg')
    axs[1, 0].scatter(population_fitness[:,2], ideal_distance,c=colors)
    axs[1, 0].set_title('Mon HP')
    axs[1, 1].scatter(population_fitness[:,3], ideal_distance,c=colors)
    axs[1, 1].set_title('Opp HP')


    putlabels(axs[0,0],population_fitness[:,0],ideal_distance)

    putlabels(axs[0,1],population_fitness[:,1],ideal_distance)
    putlabels(axs[1,0],population_fitness[:,2],ideal_distance)
    putlabels(axs[1,1],population_fitness[:,3],ideal_distance)

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='Distance from optimal point')

    plt.show()

'''

[[knockoff (Move object) 2 heatwave (Move object) 4
  thunderbolt (Move object) -2]
 [thunderbolt (Move object) -2 seismictoss (Move object) -1
  doubleedge (Move object) -1]
 [thunderbolt (Move object) -2 firefang (Move object) -1
  doubleedge (Move object) -2]
 [thunderbolt (Move object) -2 seismictoss (Move object) 2
  doubleedge (Move object) -2]
 [doubleedge (Move object) -2 firefang (Move object) -1
  doubleedge (Move object) -2]
 [thunderbolt (Move object) 2 firefang (Move object) 2
  doubleedge (Move object) -2]
 [surf (Move object) 3 firefang (Move object) -1
  thunderbolt (Move object) -1]
 [thunderbolt (Move object) -2 seismictoss (Move object) 2
  thunderbolt (Move object) -1]
 [thunderbolt (Move object) -2 firefang (Move object) -1
  thunderbolt (Move object) -1]
 [surf (Move object) 3 firefang (Move object) -1 doubleedge (Move object)
  -2]
 [thunderbolt (Move object) -2 firefang (Move object) 2
  thunderbolt (Move object) -1]
 [doubleedge (Move object) -2 firefang (Move object) 2
  doubleedge (Move object) -2]
 [knockoff (Move object) 2 firefang (Move object) -1
  thunderbolt (Move object) -1]
 [thunderbolt (Move object) 2 firefang (Move object) -1
  surf (Move object) 3]
 [thunderbolt (Move object) -2 seismictoss (Move object) 2
  thunderbolt (Move object) -2]
 [knockoff (Move object) 2 seismictoss (Move object) -1
  doubleedge (Move object) -1]
 [thunderbolt (Move object) -2 heatwave (Move object) 4
  doubleedge (Move object) -2]
 [doubleedge (Move object) 2 firefang (Move object) -1
  doubleedge (Move object) -2]
 [doubleedge (Move object) 2 seismictoss (Move object) -1
  doubleedge (Move object) -1]
 [thunderbolt (Move object) 2 seismictoss (Move object) -1
  doubleedge (Move object) -1]]
[[100.         100.           0.           0.        ]
 [  0.          30.38461538  35.02994012 100.        ]
 [  0.          24.11676647  43.07692308 100.        ]
 [ 81.30081301  50.          43.07692308  18.69918699]
 [  0.          24.11676647  43.07692308 100.        ]
 [100.         100.           0.           0.        ]
 [ 34.95934959  27.52072778  17.36526946  65.04065041]
 [ 81.30081301  63.01934592  17.36526946  18.69918699]
 [  0.          37.13611239  17.36526946 100.        ]
 [ 34.95934959  24.11676647  43.07692308  65.04065041]
 [ 70.28455285  63.01934592  17.36526946  29.71544715]
 [ 70.28455285  50.          43.07692308  29.71544715]
 [ 51.2195122   56.75149701  17.36526946  48.7804878 ]
 [ 47.15447154  32.01174574  42.10502073  52.84552846]
 [ 81.30081301  60.76923077  19.61538462  18.69918699]
 [ 51.2195122   50.          35.02994012  48.7804878 ]
 [ 81.2195122   50.          43.07692308  18.7804878 ]
 [ 95.12195122  31.03984339  43.07692308   4.87804878]
 [ 95.12195122  50.          35.02994012   4.87804878]
 [ 47.15447154  50.          35.02994012  52.84552846]]
'''

pop =[[100,         100,           0,           0,        ],
 [  0.         , 30.38461538 , 35.02994012 ,100.        ],
 [  0.         , 24.11676647 , 43.07692308 ,100.        ],
 [ 81.30081301 , 50.         , 43.07692308 , 18.69918699],
 [  0.         , 24.11676647 , 43.07692308, 100.        ],
 [100.         ,100.         ,  0.         ,  0.        ],
 [ 34.95934959 , 27.52072778 , 17.36526946,  65.04065041],
 [ 81.30081301 , 63.01934592 , 17.36526946,  18.69918699],
 [  0.         , 37.13611239 , 17.36526946, 100.        ],
 [ 34.95934959 , 24.11676647 , 43.07692308,  65.04065041],
 [ 70.28455285 , 63.01934592 , 17.36526946,  29.71544715],
 [ 70.28455285 , 50.         , 43.07692308,  29.71544715],
 [ 51.2195122  , 56.75149701 , 17.36526946,  48.7804878 ],
 [ 47.15447154 , 32.01174574 , 42.10502073,  52.84552846],
 [ 81.30081301 , 60.76923077 , 19.61538462,  18.69918699],
 [ 51.2195122  , 50.         , 35.02994012,  48.7804878 ],
 [ 81.2195122  , 50.         , 43.07692308,  18.7804878 ],
 [ 95.12195122 , 31.03984339 , 43.07692308,   4.87804878],
 [ 95.12195122 , 50.         , 35.02994012,   4.87804878],
 [ 47.15447154 , 50.         , 35.02994012,  52.84552846]]
generate_multilevel_diagram(pop)
