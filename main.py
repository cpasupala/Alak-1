import re
import numpy as np
from random import choice
from IPython import display
import matplotlib.pyplot as plt

class NeuralNetwork:
  '''
  Class NeuralNetwork
  The main datastructure of this class is another embedded class (NLayer)
  to represent a layer of the NN. This class remembers the user's specifications
  for various layers, the activation function, and training loss data.
  Also, importantly it has a list of NLayer objects to remember data specific
  to each layer (like, inputs, outputs, deltas and weights pertaining to that
  layer)
  '''

  __spec = []
  __debug = False
  __act_fp = lambda x:1/(1+np.exp(-x))
  __act_bp = lambda x:x*(1-x)
  __act_threshold = lambda x:x>0.5
  __l_data = []
  __lossfunc = []
  __plotsample = []

  class NLayer:
    '''
    This is the core of the computation. NLayer object is contains data
    pertaining to a particular layer. This includes the inputs that the layer
    receives, the weights the inputs carry, the deltas, and the output.
    Various attributes of layers are initialized in different stages of training.
    '''

    __w = []
    __id = -1
    __inp = []
    __out = []
    __delta = []


    def __init__(self,l_no,ptrons,inplen=-1):
      '''
      NLayer init : initialize the layer number and the weights provided this is
      not layer 0. layer 0 weights can be initialized only when we know the
      input size.
      Also, for now, bias is added by default. User need not provide bias.
      '''
      self.id = l_no
      if(l_no):
        self.w = np.random.random((inplen+1)*ptrons).reshape(ptrons,inplen+1)
      return

    def l_set_w(self,ptrons,inplen):
      '''
      Function to set the weights for the layer. Used particularly to set the
      weights for layer0 after we receive the input.
      '''
      self.w = np.random.random((inplen+1)*ptrons).reshape(ptrons,inplen+1)
      return

    def dump(self):
      '''
      print the data of the layer
      '''
      print(f'Layer No: {self.id}')
      print(f'shape of weights:{self.w.shape}, {self.w}')
      print(f'shape of inputs :{self.inp.shape},{self.inp}')
      print(f'shape of delta  :{self.delta.shape},{self.delta}')
      return

    def get_id(self):
      '''
      to know which layer we are in.
      '''
      return self.id

    def l_fp(self,inp,f):
      '''
      takes the input to the layer and activation function and produces the
      output and returns the same.
      '''
      inp = np.atleast_2d(inp)
      self.inp = np.column_stack((np.ones(len(inp)),inp))
      self.out = f(np.dot(self.inp,self.w.T))
      return self.out

    def l_set_delta(self, delta,g):
      '''
      Calculation of delta for the layer. It receives the delta from the
      previous layer and also the activation function's back prop definition,
      and calculates the delta for this layer and returns the calculated delta
      to be used for the next layer processing.
      '''
      delta = np.atleast_2d(delta)
      self.delta = delta * g(self.out)
      return np.dot(self.delta, self.w[:,1:])

    def l_update_w(self,alpha):
      '''
      function that updates the weights using the learning rate, its input
      and the delta. input and delta are already stored in the object.
      '''
      self.w += alpha * (np.dot(self.inp.T,self.delta)).T
      return


  def __init__(self,spec=[1],activation='sigmoid',debug=False):
    '''
    Set up the ANN framework - save the user given layer specification,
    create a list of NLayer objects to represent layers as per the specification.
    Also, initialize weights of layers other than layer0, whose input will be
    known only during fit().
    '''

    # Sanity checks
    if (any((np.array(spec)<=0)==True)):
      raise Exception('Layers can not have zero or less perceptrons !')
    if (spec[-1] != 1):
      raise Exception ('For now, single output is supported. Change the last \
      specification to contain only one perceptron')
    self.spec = np.copy(spec)
    self.debug = debug
    self.lossfunc = []
    self.l_data = []
    self.plotsample = np.random.uniform(0,1,(2000,2))
    if (activation=='sigmoid'):
      self.act_fp = lambda x:1/(1+np.exp(-x))
      self.act_bp = lambda x:x*(1-x)
      self.act_threshold = lambda x: x>0.5
    elif (activation == 'tanh'):
      self.act_fp = lambda x:(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
      self.act_bp = lambda x:1-(x**2)
      self.act_threshold = lambda x:x>0.5
    else:
      raise Exception('[NeuralNetwork:init ] Unrecognized activation function.use \'sigmoid\' or \'tanh\'')
    if (not(len(spec)>0)):
      raise Exception('[NeuralNetwork:init ] specification length can not be less than 1')

    # Setup the layer data based on the user specification.
    for i in range(len(spec)):
      if (i):
        self.l_data.append(self.NLayer(i,spec[i],spec[i-1]))
      else:
        self.l_data.append(self.NLayer(i,spec[i]))
    return

  def fp(self,inp):
    ''' Wrapper function for forward propagation. This function inturn calls
    the layer specific forward prop function with the proper input parameters
    '''
    x = inp
    for i in self.l_data:
      x = i.l_fp(x,self.act_fp)
    return x

  def bp(self, error):
    ''' Wrapper function for backward propagation. This function inturn calls
    the layer specific back prop function to calculate the deltas. This function
    inputs the delta from the previous layer in back propagation.
    '''
    delta = error
    for i in range(len(self.l_data)-1,-1,-1):
      delta = self.l_data[i].l_set_delta(delta,self.act_bp)
    return

  def updateweights(self,alpha):
    '''
    This is a wrapper function that calls the update function of each layer.
    '''
    _ = [i.l_update_w(alpha) for i in self.l_data]
    return

  def liveupdate(self,step,X,Y,live):
    val = []
    error = 0
    printstr = f'Step {step} : \n(data,prediction,expected):\n'

    for i in range(len(X)):
      pred = self.fp(X[i])
      error += (Y[i]-pred)**2
      val.append((X[i],round(pred[0][0],4),Y[i]))

    if (not live):
      plty = np.array(self.lossfunc)**2
      pltx = np.array(list(range(0,len(plty))))
      if(len(plty>100)):
        plt.plot(pltx,np.convolve(plty, np.ones((100,))/100, mode='same'))
    else:
      z = self.predict(self.plotsample)
      plt.scatter(self.plotsample[:,0],self.plotsample[:,1],c=z)
      xx = np.array(np.meshgrid(np.array([0,1]),np.array([0,1]))).T.reshape(-1,2)
      z = self.predict(xx)
      plt.scatter(xx[:,0],xx[:,1],s = 200, c=z)

    for i in range(len(val)):
      printstr += f'{val[i]}\n'
    rmserr = (error/len(Y))**0.5
    printstr += f'RMS error: {rmserr}\n'

    plt.figtext(0,0,printstr,fontsize=14)
    plt.subplots_adjust(left=0.45)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    return rmserr

  def find_rms_error(self,X,Y):
    '''
    This function calculates the rms error for the input output pairs.
    '''
    print('Results (data,prediction,expected):')
    error = 0
    for i in range(len(X)):
      val = self.fp(X[i])
      print(f'{X[i].value()}, {val}, {Y[i]}')
      error += (Y[i]-val)**2
    rmse = np.sqrt(error/len(Y))
    print(f'RMS error: {rmse}')
    return rmse

  def fit(self, X, Y, learning_rate=0.5, steps=100000, tol=1e-2,live_show=True):
    '''
    Do the training. Takes X, Y, learning_rate, iterations and
    the error tolerance. It loops over steps performing 3 steps
    1. forward propagate
    2. back propagate
    3. update weights

    set live_show to True if the display is to be the sample plot
    set live_show to False if the display is to be the training loss
    '''

    # basic sanity checks
    if (len(X)!=len(Y)):
      raise Exception ('[NeuralNetwork:fit] Length mismatch X and Y')
    if ((not(len(X)>0)) or (not(len(Y)>0))):
      raise Exception ('[NeuralNetwork:fit] X and Y should have length > 0')

    # Set up layer0 first
    self.l_data[0].l_set_w(self.spec[0],len(X[0]))

    success = False
    rmse = []
    print_interval = steps//10
    plt.figure()
    # Train
    if (print_interval==0):
      raise Exception ('steps should be atleast 10')
    for i in range(steps+1):

      # Live update the training progress
      if ((i%print_interval) == 0):
        rmse.append(self.liveupdate(i,X,Y,live_show))

      # Check for the model convergence
      if(rmse[-1] < tol):
        success = True
        self.liveupdate(i,X,Y,live_show)
        print('NN training Succeeded!')
        break
      # Model has not converged, so pick up random input and train
      # stochastic choice of training input
      idx = choice(range(len(Y)))

      # step 1 : forward propagate with the input
      z = self.fp(X[idx])
      # step 2 : Calculate the error
      error = Y[idx] - z
      # step 3 : Back propagate with the error
      self.bp(error)
      # step 4 : update the weights and repeat
      self.updateweights(learning_rate)
      self.lossfunc.append(error[0][0])

    if (rmse[-1]>=tol):
      print('NN training failed ! Tune the hyper parameters')
    if (self.debug):
      print(f'given spec is {self.spec}')
      for i in range(len(self.l_data)):
        self.l_data[i].dump()
    plt.show()
    return success

  def predict(self,x):
    return self.act_threshold(self.fp(x))

  def visual_NN_boundaries(self, Nsamp=2000):
    # generate 200 samples from uniform distribution with 2 values for each sample
    rvs = np.random.uniform(0,1,(Nsamp,2))
    # predict
    z = self.predict(rvs)
    # Visualize
    plt.title('Visualizing the ANN prediction results')
    plt.scatter(rvs[:,0],rvs[:,1],c=z)

    # plot the four corners and predict them too
    xx = np.array(np.meshgrid(np.array([0,1]),np.array([0,1]))).T.reshape(-1,2)
    z = self.predict(xx)
    plt.scatter(xx[:,0],xx[:,1],s = 200, c=z)
    plt.show()
    return


def validate_board(s,o):
    sarr = np.array((' '.join(s)).split(' '))
    if (o == 'x'):
    # Parse for the kill and then for suicide
        kill = re.compile('(?=(xo+x))')
        suicide = re.compile('(?=(ox+o))')
    elif (o=='o'):
        kill = re.compile('(?=(ox+o))')
        suicide = re.compile('(?=(xo+x))')
    else:
        raise Exception('Wrong input for offside')

    for m in re.finditer(kill,s):
        sarr[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
    s = ''.join(sarr)
    for m in re.finditer(suicide,s):
        sarr[m.start(1)+1:m.end(1)-1] = (' '.join('_'*(len(m.group(1))-2))).split(' ')
    return ''.join(sarr)



if __name__ == "__main__":
    nn = NeuralNetwork([2,2,1], activation='sigmoid',debug=False)
    x = [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]

    y = [0,1,1,0]

    s = nn.fit(x,y,learning_rate=0.2,steps=100000, tol=0.02, live_show=True)


if __name__ == "a__main__":
    s = input('input board conf: ')
    while True:
        offside = input('Offside (x or o) ?')
        if ((offside == 'x') or (offside =='o')):
            break
    out = validate_board(s,offside)
    print(out)

