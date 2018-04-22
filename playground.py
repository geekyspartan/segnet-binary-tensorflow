# Load standard packages
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import numpy as np
import scipy.io as sio

# Load submodular package
import gm_submodular
from file_operations import read_experiment_images

import gm_submodular.example_objectives as ex


# cat_dic = sio.loadmat('cat_reps.mat')
# dog_dic = sio.loadmat('dog_reps.mat')
# snake_dic = sio.loadmat('snake_reps.mat')
# monkey_dic = sio.loadmat('monkey_reps.mat')
# cat2_dic = sio.loadmat('cat2_reps.mat')
#
# cat_weights = cat_dic['weights']
# dog_weights = dog_dic['weights']
# snake_weights = snake_dic['weights']
# monkey_weights = monkey_dic['weights']
# cat2_weights = cat2_dic['weights']
#
# cat_weights_repeated = np.repeat(cat_weights,25,0)
# dog_weights_repeated = np.repeat(dog_weights,25,0)
# snake_weights_repeated = np.repeat(snake_weights,25,0)
# monkey_weights_repeated = np.repeat(monkey_weights,25,0)
# cat2_weights_repeated = np.repeat(cat2_weights,25,0)
#
# all_weights = np.concatenate((cat_weights_repeated,dog_weights_repeated,snake_weights_repeated,monkey_weights_repeated,cat2_weights_repeated),0)


all_weights,all_names = read_experiment_images("/Users/aliselmanaydin/Desktop/masks/picture_vocab/")

num_points=len(all_names)


class St(gm_submodular.DataElement):
    budget = 12

    #x=np.random.rand(num_points,2)

    x = all_weights

    dist_v = dist.pdist(x)
    Y = np.ones(num_points)

    def getCosts(self):
        return np.ones(num_points)

    def getDistances(self):
        d=dist.squareform(self.dist_v)
        return np.multiply(d,d)
S=St()

reload(ex)
reload(gm_submodular)
# Define the desired objectives
objectives=[ex.representativeness_shell(S), ex.random_shell(S)]
weights=[1,0]

# Maximize the objectives
selected_elements,score,minoux_bound=gm_submodular.leskovec_maximize(S,weights,objectives,budget=12)

gr_or_not = np.array(selected_elements) > 24

plt.figure(figsize=(12,8)) # Definition of a larger figure (in inches)
plt.scatter(S.x[:,0],S.x[:,1], c='blue', alpha=0.66, s=30, linewidths=2)
plt.scatter(S.x[selected_elements,0],S.x[selected_elements,1], c='red', s=100, alpha=0.66, linewidths=2)
print('Selected points: %s' % ' '.join(map(lambda x: str(x),selected_elements)))
plt.title('greedy k-medoids solution')


pass