#!/usr/bin/env python

import os
import getpass
import resource
import numpy as np
from scipy import stats
import pylab as P
from numba import njit
from scipy.stats.mstats import mquantiles

class Params(object):
    def __init__(self):
        self.species = ['Rats', 'Possums', 'Stoats']
        self.nYears = 10
        self.iter = 1000
        self.k = {'Rats' : 5.0, 'Possums' : 8.0, 'Stoats' : 2.2}
        self.sigma = {'Rats' : 40, 'Possums' : 80, 'Stoats' : 300}
#        self.r = {'Rats' : 3.0, 'Possums' : np.log(.75), 'Stoats' : np.log(3.0)}
        self.startDensity = {'Rats' : 1, 'Possums' : 4, 'Stoats' : .5}

        self.adultSurv = {'Rats' : np.exp(-0.79850769621), 'Possums' :  np.exp(-0.25), 
            'Stoats' : np.exp(-0.5)}
        self.adultSurvDecay = {'Rats' : 2.1, 'Possums' : 3.0, 'Stoats' : 2.5}
        self.perCapRecruit = {'Rats' : 4.5, 'Possums' : .8, 'Stoats' : 4.5}
        self.recruitDecay = {'Rats' : 1.65, 'Possums' : 1.93, 'Stoats' : 1.5}
        self.dispersalSD = {'Rats' : 300, 'Possums' : 500, 'Stoats' : 1000}
        self.sigma = {'Rats' : 40, 'Possums' : 80, 'Stoats' : 300}
        self.g0 = {'Rats' : .05, 'Possums' : 0.1, 'Stoats' : 0.02}


        baseDir = os.getenv('BROADSCALEDIR', default='.')
        if baseDir == '.':
            baseDir = '/home/dean/workfolder/projects/dean_ownerParticipation/DataResults/Results/'
        else:
            ## IF ON NESI, ADD THE PredatorFree BASE DIRECTORY
            baseDir = os.path.join(baseDir, 'DataResults', 'Results') 
        ## GET USER
        userName = getpass.getuser()
        resultsPath = os.path.join(userName, 'OwnerParticipation','ParameterCheck')
        ## PUT TOGETHER THE BASE DIRECTORY AND PATH TO RESULTS DIRECTORY 
        self.outputDataPath = os.path.join(baseDir, resultsPath)



class SimDynamics(object):
    def __init__(self, params):
        self.params = params
        ####################
        ## RUN FUNCTIONS
        self.basicdata()
        self.plotSurv()
        self.plotSurvDD()
        self.plotRecruitDD()
        self.simPopDynamics()
        self.plotSimDynamics()
        self.plotDispersal()
        self.plotPDetect()

    def basicdata(self):
        self.nSpp = len(self.params.species)
        self.kRadius = {}
        self.kArea_HA = {}
        self.kArea_KM = {}
        self.K = {}
        self.N ={}
        self.nStorage = {}
        for spp in self.params.species:
            self.kRadius[spp] = self.params.sigma[spp] * 6.0
            self.kArea_HA[spp] = (np.pi * self.kRadius[spp]**2) / 1e4
            self.kArea_KM[spp] = (np.pi * self.kRadius[spp]**2) / 1e6
            if spp == 'Stoats':
                self.K[spp] = self.kArea_KM[spp] * self.params.k[spp]
                self.N[spp] = np.random.poisson(self.kArea_KM[spp] * 
                    self.params.startDensity[spp])
            else:
                self.K[spp] = self.kArea_HA[spp] * self.params.k[spp]
                self.N[spp] = np.random.poisson(self.kArea_HA[spp] * 
                    self.params.startDensity[spp])

            self.nStorage[spp] = np.zeros((self.params.iter, self.params.nYears),
                dtype = np.int32)
            self.nStorage[spp][0, 0] = self.N[spp]
            print('spp', spp, 'kkm', self.kArea_KM[spp], 'kha', self.kArea_HA[spp],
                'k', self.K[spp], 'n', self.N[spp], 'nStor', self.nStorage[spp][0, 0])


    def plotSurv(self):
        P.figure(figsize=(9,9))
        years = np.arange(1,10, 0.01)
        for spp in self.params.species:
            prpSurv = self.params.adultSurv[spp]**years
            P.plot(years, prpSurv, label = spp, linewidth = 4.0)
        P.xlabel('Age (years)')
        P.ylabel('Mean age-specific survival')
        P.legend(loc = 'upper right')
        fname = os.path.join(self.params.outputDataPath, 'adultAgeSurv.png')
        P.savefig(fname, format='png', dpi=120)
        P.show()

    def plotSurvDD(self):
        P.figure(figsize=(15,5))
        cc = 1
        for spp in self.params.species:
            P.subplot(1, 3, cc)
            n = np.arange(.1, self.params.k[spp] + 3, 0.01)
#            print('spp', spp, 'n', n)
            if spp == 'Stoats':
                probSurv = (self.params.adultSurv[spp] * 
                    (np.exp(-n**2 / self.params.k[spp]**self.params.adultSurvDecay[spp])))
#                xN = n #/ self.kArea_KM[spp]
                xLab = spp + ' density ($km^{-2}$)'    
            else:
                probSurv = (self.params.adultSurv[spp] * 
                    (np.exp(-n**2 / self.params.k[spp]**self.params.adultSurvDecay[spp])))
#                xN = n #/ self.kArea_HA[spp]    
                xLab = spp + ' density ($ha^{-1}$)'    
            print('spp', spp, 'psurv', probSurv[:5])
            P.plot(n, probSurv, color = 'k',  linewidth = 4.0)
            P.xlabel(xLab, fontsize = 14)
            if cc == 1:
                P.ylabel('Probability of survival', fontsize = 14)
            else:
                P.ylabel('')
            cc += 1
        P.tight_layout()
        fname = os.path.join(self.params.outputDataPath, 'pSurv_DD.png')
        P.savefig(fname, format='png', dpi=120)
        P.show()

    def plotRecruitDD(self):
        P.figure(figsize=(15, 5))
        cc = 1
        for spp in self.params.species:
            P.subplot(1, 3, cc)
            n = np.arange(.1, self.params.k[spp] + 3, 0.01)
            if spp == 'Stoats':
                pMaxRec = np.exp(-n**2 / self.params.k[spp]**self.params.recruitDecay[spp])
                recRate = self.params.perCapRecruit[spp] * pMaxRec
#                xN = n / self.kArea_KM[spp]
                xLab = spp + ' density ($km^{-2}$)'   
            else:
                pMaxRec = np.exp(-n**2 / self.params.k[spp]**self.params.recruitDecay[spp])
                recRate = self.params.perCapRecruit[spp] * pMaxRec
#                xN = n / self.kArea_HA[spp]
                xLab = spp + ' density ($ha^{-1}$)' 
            P.plot(n, recRate, color = 'k',  linewidth = 4.0)
            P.xlabel(xLab, fontsize = 14)
            if cc == 1:
                P.ylabel('Per capita recruitment rate', fontsize = 14)
            else:
                P.ylabel('')
            cc += 1
        P.tight_layout()
        fname = os.path.join(self.params.outputDataPath, 'recRate_DD.png')
        P.savefig(fname, format='png', dpi=120)
        P.show()

    def simPopDynamics(self):
        for spp in self.params.species:
            for i in range(self.params.iter):
                if spp == 'Stoats':
                    self.N[spp] = np.random.poisson(self.kArea_KM[spp] * 
                        self.params.startDensity[spp])
                else:
                    self.N[spp] = np.random.poisson(self.kArea_HA[spp] * 
                        self.params.startDensity[spp])
                if self.N[spp] == 0:
                    self.N[spp] = self.params.startDensity[spp]
                for yr in range(self.params.nYears):
                    ## DO SURVIVAL
                    if spp == 'Stoats':
                        probSurv = (self.params.adultSurv[spp] * 
                            (np.exp(-self.N[spp]**2 / 
                            self.K[spp]**self.params.adultSurvDecay[spp])))
                    else:
                        probSurv = (self.params.adultSurv[spp] * 
                            (np.exp(-self.N[spp]**2 / 
                            self.K[spp]**self.params.adultSurvDecay[spp])))
                    nSurv = np.random.binomial(self.N[spp], probSurv)
                    if spp == 'Stoats':
                        pMaxRec = np.exp(-nSurv**2 / 
                            self.K[spp]**self.params.recruitDecay[spp])
                    else:
                        pMaxRec = np.exp(-nSurv**2 / 
                            self.K[spp]**self.params.recruitDecay[spp])
                    recRate = self.params.perCapRecruit[spp] * pMaxRec
                    nRec = np.random.poisson(nSurv * recRate)
                    self.N[spp] = nRec + nSurv
                    self.nStorage[spp][i, yr] = nRec + nSurv
#                    if spp == 'Stoats':
#                        print('spp', spp, 'yr', yr, 'n', self.N[spp]/self.kArea_KM[spp])
#                    else:
#                        print('spp', spp, 'yr', yr, 'n', self.N[spp]/self.kArea_HA[spp])


    def plotSimDynamics(self):
        self.years = np.arange(self.params.nYears)
        P.figure(figsize=(15, 5))
        cc = 1
        for spp in self.params.species:
            P.subplot(1, 3, cc)
            if spp == 'Stoats':
                n = self.nStorage[spp] / self.kArea_KM[spp]
            else:
                n = self.nStorage[spp] / self.kArea_HA[spp]
            meanN = np.mean(n, axis = 0)
            print('spp', spp, 'meanN', meanN)
            quants = mquantiles(n, prob = [0.025, 0.975], axis = 0)
            P.plot(self.years, meanN , color = 'k',  linewidth = 4.0)
            P.plot(self.years, quants[0], 'k--')
            P.plot(self.years, quants[1], 'k--')
            P.xlabel('Years', fontsize = 14)
            if spp == 'Stoats':
                P.ylabel('Density of ' + spp + ' ($km^{-2}$)', fontsize = 14)
            else:
                P.ylabel('Density of ' + spp + ' ($ha^{-1}$)', fontsize = 14)

            cc += 1
        P.tight_layout()
        fname = os.path.join(self.params.outputDataPath, 'simDynNonSpace.png')
        P.savefig(fname, format='png', dpi=120)
        P.show()


    def plotDispersal(self):
        nDisp = 20000
        cc = 1
        P.figure(figsize=(15, 5))
        for spp in self.params.species:
            P.subplot(1, 3, cc)
            dx = np.random.normal(0, self.params.dispersalSD[spp], nDisp)
            dy = np.random.normal(0, self.params.dispersalSD[spp], nDisp)
            dist = np.sqrt(dx**2 + dy**2)
            quants = mquantiles(dist, prob = [0.025, 0.975])
            counts, bins, patches = P.hist(dist, bins = 50, color='0.6')
            maxCounts = max(counts)
            print('max counts', maxCounts)
            P.vlines(x = quants[0], color = 'k', linewidth = 3, ymax = maxCounts, ymin = 0)
            P.vlines(x = quants[1], color = 'k', linewidth = 3, ymax = maxCounts, ymin = 0)
            P.xlabel('Dispersal distance (m) of ' + spp, fontsize = 14)
            if cc == 1:
                P.ylabel('Frequency', fontsize = 14)
            else:
                P.ylabel('')
            cc += 1
        P.tight_layout()
        fname = os.path.join(self.params.outputDataPath, 'dispersalSpp.png')
        P.savefig(fname, format='png', dpi=120)
        P.show()

    def plotPDetect(self):
        cc = 1
        P.figure(figsize=(7,6))
        for spp in self.params.species:
#            P.subplot(1, 3, cc)
            if spp == 'Stoats':
                dist = np.arange(self.params.sigma[spp] * 3.0)
                xmax = np.max(dist + 5)
            else:
                dist = np.arange(self.params.sigma[spp] * 5.0)
            pDet = self.params.g0[spp] * np.exp(- dist**2 / 2.0 / self.params.sigma[spp]**2)
            P.plot(dist, pDet, linewidth=3, label = spp)
#            P.plot(dist, pDet, color='k', linewidth=3)
#            P.xlabel('Distance (m) from trap', fontsize = 14)
#            P.ylabel('Prob. Detection of ' + spp, fontsize = 14)
            cc += 1
        P.xlabel('Distance (m) from trap', fontsize = 14)
        P.ylabel('Probability of Detection', fontsize = 14)
        P.legend(loc = 'upper right')
        P.xlim(0, xmax)
        P.ylim(0, self.params.g0['Possums'] + .0025)
        P.tight_layout()
        fname = os.path.join(self.params.outputDataPath, 'pDetectSpp.png')
        P.savefig(fname, format='png', dpi=120)
        P.show()



########            Main function
#######
def main():

    params = Params()
    simDynamics = SimDynamics(params)

    maxMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max Mem Usage', maxMem)


if __name__ == '__main__':
    main()
