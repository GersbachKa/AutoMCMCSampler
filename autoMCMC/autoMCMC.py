import numpy as np
from matplotlib import pyplot as plt


class mcmc:
    def __init__(self):
        print("Object created")

    def setLikelyhoodFunction(self,function,param_list,logLikely=True):
        self.likeFunc = function
        self.params = param_list
        self.logLike = logLikely

    def setDataset(self,data):
        self.data = data

    def start(self,iterations):
        self.N=iterations
        self._mcmc()

    def _mcmc(self):
        #import pdb; pdb.set_trace()

        param = [] #[iteration][param]
        likelyhood = []
        accept = 0

        #Generate random starting points
        randStart = []
        randScale = []
        for p in self.params:
            randStart.append(np.random.random())
            randScale.append(np.random.random())

        param.append(randStart)
        jump_scale = randScale


        #Calculate likelyhood with params
        likelyhood.append(self.likeFunc(self.data,param[0]))

        i = 0
        while i<self.N-1:
            #refineJumpScale-------------------------------
            if i==250:
                perc = accept/250
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
                testParams.append(param[i][j]+np.random.normal(0,jump_scale[j]))

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

        self.paramChains = param
        self.likelyhoods = likelyhood
        self.acceptance = (accept/self.N)*100




    def _refine_jump_scale(self,percent,jump):
        if(percent < .45):
            if percent <.35:
                #Need big shifts
                for i in range(0,len(jump)):
                    jump[i] *= (1/np.random.uniform(1,20))
                return True

            else:
                #Need smaller shifts
                for i in range(0,len(jump)):
                    jump[i] *= (1/np.random.uniform(1,10))
                return True

        elif(percent > .85):
            if percent > .95:
                #Need big shifts
                for i in range(0,len(jump)):
                    jump[i] *= np.random.uniform(1,20)
                return True

            else:
                #Need smaller shifts
                for i in range(0,len(jump)):
                    jump[i] *= np.random.uniform(1,10)
                return True
        else:
            return False

    def showChains(self):
        chains = np.swapaxes(self.paramChains,0,1)
        for i in range(0,len(self.params)):
            plt.plot(chains[i])
            plt.title(self.params[i])
            plt.show()

    def showHistograms(self,bins=100,burn=.25):
        chains = np.swapaxes(self.paramChains,0,1)
        for i in range(0,len(self.params)):
            plt.hist(chains[i][int(len(chains[i])*burn):],bins=100)
            plt.title(self.params[i])
            plt.show()
