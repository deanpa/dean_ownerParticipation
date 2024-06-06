#!/usr/bin/env python

import os
from pathlib import Path
import numpy as np
from scipy import stats
import pylab as P
from scipy.stats.mstats import mquantiles
import json
import csv
import ownerParallel
import resource

class ProcessResults(object):
    def __init__(self, params):
        self.params = params
        self.iter = 20
        ####################
        ## RUN FUNCTIONS
        self.iterWrapper()
#        self.plotCentralDensity()
#        self.plotNStorage()
#        self.plotPEradication()
#        self.plotCosts()
#        self.plotDensityCost()
#        self.plotDensityTrapArea()
#        self.plotPEradCost()

    def iterWrapper(self):
        self.years = np.arange(self.params.nYears)
        self.hrRadius = int(self.params.sigma[self.params.species] * 3)
        self.hrDiameter = self.hrRadius * 2
        self.extentSide = self.hrDiameter * self.params.extentHRMultiplier
        self.areaHa = self.extentSide**2 / 10000
        self.centralDensity = np.zeros((self.iter, self.params.nYears))
        for i in range(self.iter):
            fName = 'ownerResults_{}_Job_{}.json'.format(self.params.species, i)
            self.simResultsPath = os.path.join(self.params.outputDataPath, fName)
            self.readJSON(i)

            print('fName', fName)

        self.meanN = np.mean(self.nStorage, axis = 1)
        self.nQuants = mquantiles(self.nStorage, axis = 1, prob = [0.025, 0.975])

        self.pErad = np.sum(self.eradEventSum, axis = 1) / self.iter

        ## N PREDS IN TRAPPING AREA AT END OF OPERATION
        self.areaOutsideProp = self.areaHa - self.propArea
        densityTrapArea = self.nTrappingArea / self.areaOutsideProp[:, np.newaxis] * 100
        self.meanNTrapArea = np.mean(densityTrapArea, axis = 1)
        self.quantsNTrapArea = mquantiles(densityTrapArea, axis = 1, prob = [0.025, 0.975])


    def readJSON(self, i):
        # Load the JSON file back into a list
        fName = 'ownerResults_{}_Job_{}.json'.format(self.params.species, i)
        simResultsPath = os.path.join(self.params.outputDataPath, fName)

        with open(simResultsPath, 'r') as json_file:
            self.simResultsLists = json.load(json_file)
            
            centDen = np.array(self.simResultsLists['CentralDensity'])[-1]
            self.centralDensity[i] = centDen

        # Convert the list back to a NumPy array
        if i == 0:
            self.propRadius = np.array(self.simResultsLists['propRadius'])
            self.propCircumference = np.pi * 2.0 * self.propRadius
            self.propArea = np.array(self.simResultsLists['propArea'])
            self.paRatio = np.zeros(len(self.propArea))
            self.paRatio[1:] = self.propCircumference[1:] / self.propArea[1:]  #(m / ha)
            self.hrAreaHa = np.pi * (self.hrRadius**2) / 10000
            self.propertyHR_Ratio = self.propArea / self.hrAreaHa
            self.costs = np.array(self.simResultsLists['cost']) / self.areaHa
            self.costDenseTrapping = np.array(self.simResultsLists['costDenseTrap'])

            self.nStorage = np.expand_dims(np.array(self.simResultsLists['nStorage']) / 
                self.areaHa *100, 1)
            self.eradEventSum = np.expand_dims(np.array(
                self.simResultsLists['eradEventSum']), 1)
            self.nTrappingArea = np.expand_dims(np.array(
                self.simResultsLists['nTrappingArea']), 1)
        ## STACK ITERATIONS
        else:
            nStorage_i = np.array(self.simResultsLists['nStorage']) / self.areaHa *100
            self.nStorage = np.hstack([self.nStorage, np.expand_dims(nStorage_i, 1)])
            eradEventSum_i = np.array(self.simResultsLists['eradEventSum'])
            self.eradEventSum = np.hstack([self.eradEventSum, 
                np.expand_dims(eradEventSum_i, 1)])
            nTrappingArea_i = np.array(self.simResultsLists['nTrappingArea'])
            self.nTrappingArea = np.hstack([self.nTrappingArea, 
                np.expand_dims(nTrappingArea_i, 1)])

    def plotCentralDensity(self):
        meanCD = np.mean(self.centralDensity, axis = 0)
        quants = mquantiles(self.centralDensity, axis = 0, prob = [0.025, 0.975])
        P.figure(figsize=(8,8))
        P.plot(self.years, meanCD, color='k', linewidth=3)
        P.plot(self.years, quants[0], '--k')
        P.plot(self.years, quants[1], '--k')
        P.xlabel('Years', fontsize = 16)
        P.ylabel('Density ' + '($ha^{-1}$)', fontsize = 16)
        fname = 'centralDensity_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        P.show()

    def plotNStorage(self):
#        self.nStorage = np.array(self.simResultsLists['nStorage']) / self.areaHa *100
#        meanN = np.mean(self.nStorage, axis = 0)
#        quants = mquantiles(self.nStorage, axis = 0, prob = [0.025, 0.975])
        ## PLOT PA RATIO
        P.figure(figsize=(8,8))
        P.plot(self.paRatio[1:], self.meanN[1:], color='b', linewidth=3)
        P.fill_between(self.paRatio[1:], self.nQuants[1:, 0], self.nQuants[1:, 1], 
            alpha = 0.2, color = 'b', lw = 0)
#        P.plot(self.paRatio[1:], self.nQuants[0,1:], '--k')
#        P.plot(self.paRatio[1:], self.nQuants[1, 1:], '--k')
        P.xlabel('Ratio of property perimeter to area', fontsize = 16)
        P.ylabel('Density ' + '($km^{2}$)', fontsize = 16)
        fname = 'density_paRatio_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        P.show()

        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(8,8))
        P.plot(self.propertyHR_Ratio, self.meanN, color='b', linewidth=3)
        P.fill_between(self.propertyHR_Ratio, self.nQuants[:, 0], self.nQuants[:, 1], 
            alpha = 0.2, color = 'b')
#        P.plot(self.propertyHR_Ratio, self.nQuants[0], '--k')
#        P.plot(self.propertyHR_Ratio, self.nQuants[1], '--k')
        P.xlabel('Ratio of property area to HR area', fontsize = 16)
        P.ylabel('Density ' + '($km^{2}$)', fontsize = 16)
        fname = 'den_Prop_HR_Ratio_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        P.show()


    def plotPEradication(self):
#        eradEventSum = np.array(self.simResultsLists['eradEventSum'])
        print('pErad', self.pErad)
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(8,8))
        P.plot(self.propertyHR_Ratio, self.pErad, color='k', linewidth=3)
        P.axhline(y = 0.95, color = 'k', linestyle = 'dashed')

        P.xlabel('Ratio of property area to HR area', fontsize = 16)
        P.ylabel('Probability of eradication', fontsize = 16)
        fname = 'pEradication_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        P.show()


    def plotCosts(self):
#        costs = np.array(self.simResultsLists['cost']) / self.areaHa
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(8,8))
        P.plot(self.propertyHR_Ratio, self.costs, color='k', linewidth=3)
        P.xlabel('Ratio of property area to HR area', fontsize = 16)
        P.ylabel('Trapping costs (\$$ ha^{-1}$)', fontsize = 16)
        fname = 'trappingCosts_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        P.show()

        ## PLOT PROPERTY PERIMETER IN METRES VS COST
        P.figure(figsize=(8,8))
        P.plot(self.propCircumference, self.costDenseTrapping, color='k', linewidth=3)
        P.xlabel('Property circumference (m)', fontsize = 16)
        P.ylabel('Total trapping costs in buffer area ($)', fontsize = 16)
        fname = 'trapCosts_Perimeter_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        P.show()
        print('Cost in dense area per m of perimeter', 
            self.costDenseTrapping[1:] / self.propCircumference[1:],
            'Circumference', self.propCircumference[1:])



    def plotDensityCost(self):
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(8,8))
        ax1 = P.gca()
        ax1.plot(self.propertyHR_Ratio, self.meanN, color='b', linewidth=3)
        ax1.fill_between(self.propertyHR_Ratio, self.nQuants[:, 0], self.nQuants[:, 1], 
            alpha = 0.2, color = 'b')
#        P.plot(self.propertyHR_Ratio, self.nQuants[0], '--k')
#        P.plot(self.propertyHR_Ratio, self.nQuants[1], '--k')
        ax2 = ax1.twinx()
        ax2.plot(self.propertyHR_Ratio, self.costs, color='k', linewidth=3)


        ax1.set_xlabel('Ratio of property area to HR area', fontsize = 16)
        ax1.set_ylabel('Density ' + '($km^{2}$)', fontsize = 16)
        ax1.spines["right"].set_edgecolor("blue")
        ax1.tick_params(axis='y', colors="blue")
        ax1.yaxis.label.set_color("blue")


        ax2.set_ylabel('Trapping costs (\$$ ha^{-1}$)', fontsize = 16)


        fname = 'den_Cost_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        P.show()



    def plotDensityTrapArea(self):
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(8,8))
        ax1 = P.gca()
        ax1.plot(self.propertyHR_Ratio, self.meanNTrapArea, color='b', linewidth=3)
        ax1.fill_between(self.propertyHR_Ratio, self.quantsNTrapArea[:,0], 
            self.quantsNTrapArea[:, 1], alpha = 0.2, color = 'b', lw = 0)
        ax2 = ax1.twinx()
        ax2.plot(self.propertyHR_Ratio, self.costs, color='k', linewidth=3)
        ax1.set_xlabel('Ratio of property area to HR area', fontsize = 16)
        ax1.set_ylabel('Density in trapped area ' + '($km^{2}$)', fontsize = 16)
        ax1.spines["right"].set_edgecolor("blue")
        ax1.tick_params(axis='y', colors="blue")
        ax1.yaxis.label.set_color("blue")
        ax2.set_ylabel('Trapping costs (\$$ ha^{-1}$)', fontsize = 16)
        fname = 'denTrapArea_Cost_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        P.show()


    def plotPEradCost(self):
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(8,8))
        ax1 = P.gca()
        ax1.plot(self.propertyHR_Ratio, self.pErad, color='b', linewidth=3)
        ax1.axhline(y = 0.95, color = 'b', linestyle = 'dashed')
        ax2 = ax1.twinx()
        ax2.plot(self.propertyHR_Ratio, self.costs, color='k', linewidth=3)
        ax1.set_xlabel('Ratio of property area to HR area', fontsize = 16)
        ax1.set_ylabel('Probability of eradication', fontsize = 16)
        ax1.spines["right"].set_edgecolor("blue")
        ax1.tick_params(axis='y', colors="blue")
        ax1.yaxis.label.set_color("blue")
        ax2.set_ylabel('Trapping costs (\$$ ha^{-1}$)', fontsize = 16)
        fname = 'pErad_Cost_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        P.show()




########            Main function
#######
def main():
    #foo2(2,12,10)

    params = ownerParallel.Params()
    processresults = ProcessResults(params)

    maxMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max Mem Usage', maxMem)


if __name__ == '__main__':
    main()
