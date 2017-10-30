# What is a Restricted Boltzmann Machine(RBM)
* Restricted Boltzmann Machines - Ep. 6 (Deep Learning SIMPLIFIED): https://youtu.be/puux7KZQfsE
* Deep Belief Nets - Ep. 7 (Deep Learning SIMPLIFIED): https://youtu.be/E2Mt_7qked0
* Boltzmann Machine Deep Learning A-Z Udemy : http://bit.ly/2yWF6eP
* How a Boltzmann machine models data by Geoffrey Hinton : https://youtu.be/kytxEr0KK7Q
* Restricted Boltzmann Machines by Geoffrey Hinton : https://youtu.be/EZOpZzUKl48
* An example of RBM learning by Geoffrey Hinton  : https://youtu.be/iHaS6O1eox4
* Generate Music in TensorFlow Siraj Raval (with RBM) : https://youtu.be/ZE7qWXX05T0

# RBM Model for movie recommender 

```python
### Part 1 : Creating the architecture of the Neural Network 

# Create the class RBM 
class RBM(): 
# self is the object 
# all the variables attached to the object will be created with self. 
# nv number of visible nodes 
# nh number of hidden nodes 
def __init__(self, nv, nh): 
 	#initialize the parameters we optimize during the training weights and bias
 	#weights used for the probability of the visible nodes given the hidden nodes (p_v_given_h))
 	# torch.rand : random normal distribution mean=0, variance=1 
 	self.W = torch.randn(nh,nv)
 	# bias probability of the hidden nodes given the visible nodes (p_h_given_v))
 	# fake dimension for the batch = 1
 	self.a = torch.randn(1,nh)
 	# bias probability of the visible nodes is activated 
 	#given the value of the hidden nodes (p_v_given_h))
 	self.b = torch.randn(1, nv)


def sample_h(self, x): 
	# probability h is activated given the value v is the sigmoid(Wx+a).
	# torch.mm make the product of two tensors. 
	# W.t()take the transpose because W is used for the p_v_given_h.
	wx=torch.mm(x,self.W.t())
	# .expand_as(wx) : expand the mini-batch.
	activation=wx+self.a.expand_as(wx)
	# probability p_h_given_v is the probability that the note drama genre is activated. 
	# v value is the input value. If v is a film drama, p_h_given_v will be hight. 
	# If v is not a film drama, p_h_given_v will be low.
	p_h_given_v=torch.sigmoid(activation)
	# Bernouilli RBM. we predict the user loves the movie or not (0 or 1).
	# activation or not activation of the nh neurons. 
	return p_h_given_v, torch.bernouilli(p_h_given_v)


def sample_v(self, y): 
	# probability h is activated given the value v is the sigmoid(Wx+a).
	# torch.mm make the product of two tensors. 
	wy=torch.mm(y,self.W)
	# .expand_as(wx) : expand the mini-batch.
	activation=wy+self.b.expand_as(wy)
	p_v_given_h=torch.sigmoid(activation)
	# Bernouilli RBM. we predict the user loves the movie or not (0 or 1).
	# activation or not activation of the nv neurons. 
	return p_v_given_h, torch.bernouilli(p_v_given_h)

# Contrastive divergence Algorithm
# Optimize the weights to minimize the energy.
# ~ Maximize the Log-Likelihood of the model. 
# Need to approximate the gradients with the algorithm contrastive divergence. 
def train(self,v0,vk,ph0,phk): 
	self.W += torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)
	# add ,0 for the tensor of two dimension 
	self.b += torch.sum((v0-vk),0)
	self.a += torch.sum(ph0-phk,0)

### Part 2 : Create the RBM Object 
# number of movies 
nv=len(training_set[0]) 
# parameter is tunable is the number of features that we want to detect 
# features ~ genre, actors, director, oscar, date.... 
nh=100 
# update the weights after serveral observations, also tunable
batch_size=100
# Creation of the object of the class RBM()
rbm=RBM(nv,nh)

### Part 3 : Training the RBM 
nb_epoch = 10 
# upper bound is no included nb_epoch+1 

# First for loop : epoch for loop 
for epoch in range (1,nb_epoch+1): 
	#loss function initialized to 0 at the beginning of the trainning 
	train_loss = 0 
	# counter which is a float . 
	s = 0.
	

	# Second for loop : user forloop 
	# 0 lower bound 
	# nb_users-batch_size upper bound 
	# batch_size is the step of each batch (100)
	# First batch is from user id=0 ti user id =99
	for id_user in range(0,nb_users-batch_size,batch_size): 
		# at the beginning v0=vk 
		# vk is going to be updated 
		# id_user,id_user+batch_size ~id_user+100
		vk=training_set[id_user,id_user+batch_size]
		v0=training_set[id_user,id_user+batch_size]
		ph0, _ = rbm.sample_h(v0)
		
		# Third for loop : Contrastive divergence
		for k in range(10): 
			_,hk=rbm.sample_h(vk)
			_,vk=rbm.sample_v(hk)
			# we don't want to learn where is no rating by the user
			# no update when -1 rating. 
			vk[v0<0]=v0[v0<0]
			phk,_=rbm.sample_h(vk)
			rbm.train(v0,vk,ph0,phk)
			# Compare vk updated after the training to v0 the target. 
			# simple distance in absolute value 
			# [vO>=0] take only the value with ratings / coherence with vk[v0<0]=[v0<0]
			train_loss+=torch.mean(torch.abs(v0[vO>=0]-vk[vO>=0]))
			s += 1.
print('epoch: ' +str(epoch) +' loss: '+str(train_loss/s))
```
