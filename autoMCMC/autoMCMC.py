
import sys
import numpy as np
from matplotlib import pyplot as plt


class mcmc:

#Setup Functions----------------------------------------------------------------

    def __init__(self):
        self.jumpSet = False
        self.boundsSet = False

    def setLikelyhoodFunction(self,function,param_list,logLikely=True):
        self.likeFunc = function
        self.params = param_list
        self.logLike = logLikely

    def setDataset(self,data):
        self.data = data

    def setJumpscale(self,scales):
        self.jumpSet = True
        self.scale = scales

    def setBoundaries(self,paramBoundaries):
        self.bounds = paramBoundaries
        self.boundsSet = True

    def start(self,iterations):
        self.N=iterations
        self._mcmc()

#Actual mcmc--------------------------------------------------------------------
    def _mcmc(self):
        param = [] #[iteration][temp][param]
        likelyhood = []
        accept = 0

        #Generate random starting points
        if not self.boundsSet:
            #No boundaries
            param.append(np.random.random((len(self.params))))
        else:
            #Boundaries are set
            randStart = []
            for par in range(len(self.params)):
                randStart.append(np.random.uniform(self.bounds[0],self.bounds[1]))
            param.append(randStart)


        if not self.jumpSet:
            jump_scale = np.random.random(len(self.params))
        else:
            jump_scale = self.scale


        #Calculate likelyhood with params
        likelyhood.append(self.likeFunc(self.data,param[0]))

        i = 0
        while i<self.N-1:
            #Progress Bar-------------------------------------------------------
            if i>300:
                percentComplete = int((i/self.N)*100)
                progressStr = ''
                for p in range(10,110,10):
                    if percentComplete >= p:
                        progressStr+='#'
                    else:
                        progressStr+='-'
                print('\rProgress: [{}],{}%'.format(progressStr,percentComplete),end='')
            #refineJumpScale-------------------------------
            if i==300 and not self.jumpSet:
                perc = accept/299
                if (self._refine_jump_scale(perc,jump_scale)):
                    #True if needs to reset chain
                    i=0
                    accept=0
                    param = [param[0]]
                    likelyhood = [likelyhood[0]]

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

            #Calc likelyhood
            newLikelyhood = self.likeFunc(self.data,testParams)
            if self.logLike:
                #Loglikelyhood Function
                logH = newLikelyhood-likelyhood[i]
                if(logH>0 or logH>=np.log(np.random.random())):
                    accept+=1
                    param.append(testParams)
                    likelyhood.append(newLikelyhood)
                else:
                    param.append(param[i])
                    likelyhood.append(likelyhood[i])

            else:
                #Regular Likelyhood
                h = newLikelyhood/likelyhood[i]
                if(h>=np.random.random()):
                    accept+=1
                    param.append(testParams)
                    likelyhood.append(newLikelyhood)
                else:
                    param.append(param[i])
                    likelyhood.append(likelyhood[i])
            i+=1

        print("\rDone                      ")
        self.paramChains = param
        self.likelyhoods = likelyhood
        self.acceptance = (accept/self.N)*100

#Auxiliary functions------------------------------------------------------------
    def _refine_jump_scale(self,percent,jump):
        if(percent < .30):
            if percent <.20:
                #Need big shifts
                for i in range(0,len(jump)):
                    jump[i] *= (1/np.random.uniform(1,20))
                print("\rJump scales are too big, changing them to: {}".format(jump), end='')
                return True

            else:
                #Need smaller shifts
                for i in range(0,len(jump)):
                    jump[i] *= (1/np.random.uniform(1,10))
                print("\rJump scales are too large, changing them to: {}".format(jump), end='')
                return True

        elif(percent > .75):
            if percent > .85:
                #Need big shifts
                for i in range(0,len(jump)):
                    jump[i] *= np.random.uniform(1,20)
                print("\rJump scales are too small, changing them to: {}".format(jump), end='')
                return True

            else:
                #Need smaller shifts
                for i in range(0,len(jump)):
                    jump[i] *= np.random.uniform(1,10)
                print("\rJump scales are too small, changing them to: {}".format(jump), end='')
                return True
        else:
            print("\rJump scales have been finalized: {}            ".format(jump), end='\n')
            return False


    def _swapProposal(param_val_list):
        pass


#Visualizing--------------------------------------------------------------------
    def showChains(self):
        try:
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
            chains = np.swapaxes(self.paramChains,0,1)
            for i in range(0,len(self.params)):
                plt.hist(chains[i][int(len(chains[i])*burn):],bins=100)
                plt.title(self.params[i])
                plt.show()
        except:
            print("No chains found. Did you run \"start()\"?")
            return None

    def showLikelyhoods(self):
        try:
            plt.plot(self.likelyhoods)
            plt.title("Likelyhoods")
            plt.show()
        except:
            print("No chains found. Did you run \"start()\"?")
            return None
