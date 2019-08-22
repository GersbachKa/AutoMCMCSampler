
import sys
import numpy as np
from matplotlib import pyplot as plt


class mcmc:

#Setup Functions----------------------------------------------------------------

    def __init__(self):
        """Initiallizes a Markov-Chain Monte Carlo object.
        In order to use this object, call setLikelihoodFunction() and optionally
        call setDataset(),setJumpscale(),setBoundaries(),setTemperatures(), then
        call start().
        """
        self.jumpSet = False
        self.boundsSet = False
        self.tempSet = False
        self.data = None
        self.likeFunc = None
        self.params = None
        self.logLike = True
        self.jumpScale = None
        self.bounds = None
        self.temps = None

    def setLikelihoodFunction(self,function,param_list,logLikely=True):
        """Sets the function used in the MCMC sampler.
        In order for this to work, you need the likelihood function to accept two
        parameters: dataset which is a list of datapoints, and parameterList
        which is a list of parameter values in the same order as the next input,
        param_list is a list of parameter names, all of which are strings. The
        logLikely boolean is for specifying if the likelihood function is in log
        scale or linear scale.
        """
        self.likeFunc = function
        self.params = param_list
        self.logLike = logLikely

    def setDataset(self,data):
        """Sets the dataset used for the MCMC function.
        This is not a required function if your likelihood function does not use it.
        """
        self.data = data

    def setJumpscale(self,scales):
        """Sets the scale of the random draws of a gaussian that are used in chosing
        the next location.
        scales needs to be a list of floats or doubles which has the same length
        as the amount of variables used in setLikelihoodFunction(). Scale[i]
        corresponds to the jump scale for param_list[i]
        """
        self.jumpSet = True
        self.jumpScale = scales

    def setBoundaries(self,paramBoundaries):
        """An optional function which sets the boundaries of each parameters.
        paramBoundaries needs to be a list of touples of floats or doubles. The
        first in the touple represents the lower bound and the second represents
        the upper bound. paramBoundaries[i] corresponds to the boundaries for
        param_list[i]
        """
        self.bounds = paramBoundaries
        self.boundsSet = True

    def setTemperatures(self,temp_list):
        """A function used to set the temperatures of each chain in a parallel
        tempering MCMC chain.
        temp_list is a list of floats or doubles which corresponds to the temperature
        of each chain. The first value MUST be 1. Closer chains swap more
        """
        self.tempSet = True
        self.temps = temp_list

    def start(self,iterations,parallel_tempering=False):
        """A function used to start the MCMC algorithm.
        iterations represent how many iterations the MCMC function must go through
        before stopping and returning. Use parallel_tempering=True if you plan to
        use parallel tempering
        """
        self.pt = parallel_tempering
        if not parallel_tempering:
            self._mcmc(iterations)
        else:
            self._ptmcmc(iterations)

#Regular mcmc--------------------------------------------------------------------
    def _mcmc(self,N):
        param = [] #[iteration][param][temp]
        likelihood = []
        accept = 0

        #Generate random starting points
        if not self.boundsSet:
            #No boundaries
            param.append(np.random.random((len(self.params))))
        else:
            #Boundaries are set
            randStart = []
            for par in range(len(self.params)):
                randStart.append(np.random.uniform(self.bounds[par,0],self.bounds[par,1]))
            param.append(randStart)


        if not self.jumpSet:
            jump_scale = np.random.random(len(self.params))
        else:
            jump_scale = self.jumpScale


        #Calculate likelihood with params
        likelihood.append(self.likeFunc(self.data,param[0]))

        i = 0
        while i<N-1:
            #Progress-------------------------------------------------------
            if i>305:
                percentComplete = (int((i/N)*10000))/100
                print('\rProgress: {}%   '.format(percentComplete),end='')
            #refineJumpScale-------------------------------
            if i==300 and not self.jumpSet:
                perc = accept/299
                if (self._refine_jump_scale(perc,jump_scale)):
                    #True if needs to reset chain
                    i=0
                    accept=0
                    param = [param[0]]
                    likelihood = [likelihood[0]]

            #MCMC Part-------------------------------------
            #Get new value
            testParams = []
            for j in range(0,len(self.params)):
                next = None
                if not self.boundsSet:
                    next = param[i][j]+np.random.normal(0,jump_scale[j])
                else:
                    while next==None or next<self.bounds[j][0] or next>self.bounds[j][1]:
                        next = param[i][j]+np.random.normal(0,jump_scale[j])

                testParams.append(next)

            #Calc likelihood
            newlikelihood = self.likeFunc(self.data,testParams)
            if self.logLike:
                #Loglikelihood Function
                logH = newlikelihood-likelihood[i]
                if(logH>=np.log(np.random.uniform(0,1))):
                    accept+=1
                    param.append(testParams)
                    likelihood.append(newlikelihood)
                else:
                    param.append(param[i])
                    likelihood.append(likelihood[i])

            else:
                #Regular likelihood
                h = newlikelihood/likelihood[i]
                if(h>=np.random.random()):
                    accept+=1
                    param.append(testParams)
                    likelihood.append(newlikelihood)
                else:
                    param.append(param[i])
                    likelihood.append(likelihood[i])
            i+=1

        print("\rDone              ")
        self.jumpScale = jump_scale
        self.paramChains = param
        self.likelihoods = likelihood
        self.acceptance = (accept/N)*100

#PTMCMC-------------------------------------------------------------------------
    def _ptmcmc(self,N):
        params = np.zeros((N,len(self.params),len(self.temps))) #[iteration,parameter,temp]
        likelihoods = np.zeros((N,len(self.temps))) #[iteration,temp]
        jump_scale = []
        acceptance = np.zeros(len(self.temps)) #[temp]
        swapCount = 0

        if not self.boundsSet:
            #No boundaries
            params[0] = np.random.random((len(self.params),len(self.temps)))
        else:
            #Boundaries are set
            for par in range(len(self.params)):
                params[0,par] = np.random.uniform(self.bounds[par,0],
                                                  self.bounds[par,1],
                                                  len(self.temps))

        #Calculate likelihood with params
        for t in range(len(self.temps)):
            likeRet = self.likeFunc(self.data,params[0,:,t])
            if self.logLike:
                likelihoods[0,t] = (likeRet)*(1/self.temps[t])
            else:
                if likeRet < 0:
                    likelihoods[0,t] = -1*((-1*likeRet)**(1/self.temps[t]))
                else:
                    likelihoods[0,t] = (likeRet)**(1/self.temps[t])

        if not self.jumpSet:
            jump_scale = np.random.random(len(self.params))
            self._mcmc(302) #Already implemented in code for Single chain.
            jump_scale = self.jumpScale
        else:
            jump_scale = self.jumpScale

        i = 0
        while i < N-1:
            percentComplete = (int((i/N)*10000))/100
            print('\rProgress: {}%   '.format(percentComplete),end='')
            for t in range(len(self.temps)):
                #New location to test
                testVals = []
                for p in range(len(self.params)):
                    if self.boundsSet:
                        temp = params[i,p,t] + np.random.normal(0,jump_scale[p])
                        #If not within range set by boundaries
                        while temp < self.bounds[p,0] or temp > self.bounds[p,1]:
                            temp = params[i,p,t] + np.random.normal(0,jump_scale[p])
                        testVals.append(temp)
                    else:
                        testVals.append(params[i,p,t] + np.random.normal(0,jump_scale[p]))

                #Calculate likelihood
                likeRet = self.likeFunc(self.data,testVals)
                newlikely = None
                if self.logLike:
                    newlikely = (likeRet)*(1/self.temps[t])
                else:
                    if likeRet < 0:
                        newlikely = -1*((-1*likeRet)**(1/self.temps[t]))
                    else:
                        newlikely = ((likeRet)**(1/self.temps[t]))

                if self.logLike:
                    #Loglikelihood Function
                    logH = newlikely-likelihoods[i,t]
                    if logH>=np.log(np.random.uniform(0,1)):
                        acceptance[t]+=1
                        params[i+1,:,t] = testVals
                        likelihoods[i+1,t] = newlikely
                    else:
                        params[i+1,:,t] = params[i,:,t]
                        likelihoods[i+1,t] = likelihoods[i,t]

                else:
                    #Regular likelihood
                    h = newlikely/likelihoods[i,t]
                    if h>=np.random.random():
                        acceptance[t]+=1
                        params[i+1,:,t] = testVals
                        likelihoods[i+1,t] = newlikely
                    else:
                        params[i+1,:,t] = params[i,:,t]
                        likelihoods[i+1,t] = likelihoods[i,t]



            if i%10==0 and i!=0:
                #Chain swapping
                for t in range(len(self.temps)-1):
                    chain1Par = params[i,:,t]
                    chain2Par = params[i,:,t+1]

                    swapTop = ((abs(self.likeFunc(self.data,chain2Par))**(1/self.temps[t]))
                              *(abs(self.likeFunc(self.data,chain1Par))**(1/self.temps[t+1])))

                    swapBot = ((abs(self.likeFunc(self.data,chain1Par))**(1/self.temps[t]))
                              *(abs(self.likeFunc(self.data,chain2Par))**(1/self.temps[t+1])))

                    swapH = swapTop/swapBot

                    if swapH >= np.random.random():
                        #swap parameters
                        params[i,:,t] = chain2Par
                        params[i,:,t+1] = chain1Par

                        #swap likelihoods
                        likeRet = self.likeFunc(self.data,chain2Par)
                        if self.logLike:
                            likelihoods[i,t] = (likeRet)*(1/self.temps[t])
                        else:
                            if likeRet < 0:
                                likelihoods[i,t] = -1*((-1*likeRet)**(1/self.temps[t]))
                            else:
                                likelihoods[i,t] = ((likeRet)**(1/self.temps[t]))

                        likeRet = self.likeFunc(self.data,chain1Par)
                        if likeRet < 0:
                            likelihoods[i,t+1] = -1*((-1*likeRet)**(1/self.temps[t+1]))
                        else:
                            likelihoods[i,t+1] = ((likeRet)**(1/self.temps[t+1]))

                        swapCount+=1

            i+=1

        print("\rDone              ")
        self.swapCount = swapCount
        self.jumpScale = jump_scale
        self.paramChains = params
        self.likelihoods = likelihoods
        self.acceptance = (acceptance/N)*100






#Auxiliary functions------------------------------------------------------------
    def _refine_jump_scale(self,percent,jump):
        if(percent < .55):
            if percent <.30:
                #Need big shifts
                for i in range(0,len(jump)):
                    jump[i] *= (1/np.random.uniform(1,5))
                print("\rJump scales are too large, changing them to: {}".format(jump), end='')
                return True

            else:
                #Need smaller shifts
                for i in range(0,len(jump)):
                    jump[i] *= (1/np.random.uniform(1,2))
                print("\rJump scales are too large, changing them to: {}".format(jump), end='')
                return True

        elif(percent > .85):
            if percent > .95:
                #Need big shifts
                for i in range(0,len(jump)):
                    jump[i] *= np.random.uniform(1,5)
                print("\rJump scales are too small, changing them to: {}".format(jump), end='')
                return True

            else:
                #Need smaller shifts
                for i in range(0,len(jump)):
                    jump[i] *= np.random.uniform(1,2)
                print("\rJump scales are too small, changing them to: {}".format(jump), end='')
                return True
        else:
            print("\rJump scales have been finalized: {}               ".format(jump), end='\n')
            return False


#Visualizing--------------------------------------------------------------------
    '''
    If chain_number == None, then show all Chains
    '''
    def showChains(self,chain_number=0):
        try:
            if self.pt:
                if chain_number==None:
                    #Show all
                    for par in range(0,len(self.params)):
                        chains = self.paramChains[:,par,:]
                        for i in range(len(self.temps)):
                            plt.plot(chains[:,i])
                        plt.title(self.params[par])
                        plt.show()

                else:
                    for par in range(0,len(self.params)):
                        chains = self.paramChains[:,par,chain_number]
                        plt.plot(chains)
                        plt.title(self.params[par])
                        plt.show()
            else:
                chains = np.swapaxes(self.paramChains,0,1)
                for i in range(0,len(self.params)):
                    plt.plot(chains[i])
                    plt.title(self.params[i])
                    plt.show()
        except:
            print("No chains found. Did you run \"start()\"?")
            return None

    def showHistograms(self,bins=100,burn=.25):
        try:
            if self.pt:
                chains = np.swapaxes(self.paramChains[:,:,0],0,1)
            else:
                chains = np.swapaxes(self.paramChains,0,1)

            for i in range(0,len(self.params)):
                plt.hist(chains[i][int(len(chains[i])*burn):],bins=100)
                plt.title(self.params[i])
                plt.show()
        except:
            print("No chains found. Did you run \"start()\"?")
            return None

    def showlikelihoods(self):
        try:
            plt.plot(self.likelihoods)
            plt.title("likelihoods")
            plt.show()
        except:
            print("No chains found. Did you run \"start()\"?")
            return None
