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
        self.iter = 200
        ####################
        ## RUN FUNCTIONS
        self.iterWrapper()
        self.summaryPCapture()
        self.find_Den_PropHR()
        self.plotCentralDensity()
        self.plotNStorage()
        self.plotPEradication()
        self.plotCosts()
#        self.plotDensityCost()
#        self.plotDensityTrapArea()
        self.plot2RowsDensityCost()
#        self.plotPEradCost()

    def iterWrapper(self):
        self.years = np.arange(self.params.nYears)
        self.allSpp = ['Rats', 'Possums', 'Stoats']
        self.nSpp = len(self.allSpp)
        self.makeDict()

        for spp in self.allSpp:
            self.hrRadius[spp] = int(self.params.sigma[spp] * 3)
            self.hrDiameter[spp] = self.hrRadius[spp] * 2
            self.extentSide[spp] = self.hrDiameter[spp] * self.params.extentHRMultiplier
            self.areaHa[spp] = self.extentSide[spp]**2 / 10000
            self.centralDensity[spp] = np.zeros((self.iter, self.params.nYears))
            for i in range(self.iter):
                fName = 'ownerResults_{}_Job_{}.json'.format(spp, i)
                self.simResultsPath = os.path.join(self.params.outputDataPath, fName)
                self.readJSON(i, spp)
    
            self.meanN[spp] = np.mean(self.nStorage[spp], axis = 1)
            self.nQuants[spp] = mquantiles(self.nStorage[spp], axis = 1, prob = [0.025, 0.975])
    
            self.pErad[spp] = np.sum(self.eradEventSum[spp], axis = 1) / self.iter

            ## N PREDS IN TRAPPING AREA AT END OF OPERATION
            self.areaOutsideProp[spp] = self.areaHa[spp] - self.propArea[spp]
            densityTrapArea = self.nTrappingArea[spp] / self.areaOutsideProp[spp][:, np.newaxis] * 100
            self.meanNTrapArea[spp] = np.mean(densityTrapArea, axis = 1)
            self.quantsNTrapArea[spp] = mquantiles(densityTrapArea, axis = 1, prob = [0.025, 0.975])


    def readJSON(self, i, spp):
        # Load the JSON file back into a list
        fName = 'ownerResults_{}_Job_{}.json'.format(spp, i)
        simResultsPath = os.path.join(self.params.outputDataPath, fName)

        with open(simResultsPath, 'r') as json_file:
            self.simResultsLists = json.load(json_file)
            
            centDen = np.array(self.simResultsLists['CentralDensity'])[-1]
            self.centralDensity[spp][i] = centDen

        # Convert the list back to a NumPy array
        if i == 0:
            self.propRadius[spp] = np.array(self.simResultsLists['propRadius'])
            self.propCircumference[spp] = np.pi * 2.0 * self.propRadius[spp]
            self.propArea[spp] = np.array(self.simResultsLists['propArea'])
            self.paRatio[spp] = np.zeros(len(self.propArea[spp]))
            self.paRatio[spp][1:] = self.propCircumference[spp][1:] / self.propArea[spp][1:]  #(m / ha)
            self.hrAreaHa[spp] = np.pi * (self.hrRadius[spp]**2) / 10000
            self.propertyHR_Ratio[spp] = self.propArea[spp] / self.hrAreaHa[spp]
            self.costs[spp] = np.array(self.simResultsLists['cost']) / self.areaHa[spp]
            self.costDenseTrapping[spp] = np.array(self.simResultsLists['costDenseTrap'])
            self.pCapture[spp] = np.array(self.simResultsLists['pCapture'])

            self.nStorage[spp] = np.expand_dims(np.array(self.simResultsLists['nStorage']) / 
                self.areaHa[spp] *100, 1)
            self.eradEventSum[spp] = np.expand_dims(np.array(
                self.simResultsLists['eradEventSum']), 1)
            self.nTrappingArea[spp] = np.expand_dims(np.array(
                self.simResultsLists['nTrappingArea']), 1)

        ## STACK ITERATIONS
        else:
            nStorage_i = np.array(self.simResultsLists['nStorage']) / self.areaHa[spp] *100
            self.nStorage[spp] = np.hstack([self.nStorage[spp], np.expand_dims(nStorage_i, 1)])
            eradEventSum_i = np.array(self.simResultsLists['eradEventSum'])
            self.eradEventSum[spp] = np.hstack([self.eradEventSum[spp], 
                np.expand_dims(eradEventSum_i, 1)])
            nTrappingArea_i = np.array(self.simResultsLists['nTrappingArea'])
            self.nTrappingArea[spp] = np.hstack([self.nTrappingArea[spp], 
                np.expand_dims(nTrappingArea_i, 1)])

    def makeDict(self):
        self.hrRadius = {}
        self.hrDiameter = {}
        self.extentSide = {}
        self.areaHa = {}
        self.centralDensity = {}
        self.meanN = {}
        self.nQuants = {}
        self.pErad = {}
        self.areaOutsideProp = {}
        self.meanNTrapArea = {}
        self.quantsNTrapArea = {}
        self.centralDensity = {}
        
        self.propRadius = {}
        self.propCircumference = {}
        self.propArea = {} 
        self.paRatio = {}
        self.paRatio = {}
        self.hrAreaHa = {}
        self.propertyHR_Ratio = {}
        self.costs = {}
        self.costDenseTrapping = {}

        self.nStorage = {} 
        self.eradEventSum = {}
        self.nTrappingArea = {}
        self.pCapture = {}

    def summaryPCapture(self):
        for spp in self.allSpp:
            meanPCapt = np.mean(self.pCapture[spp])
            quantsPCapt = mquantiles(self.pCapture[spp], prob=[0.05, 0.95])
            print('Species:', spp, 'Mean PCapt', meanPCapt, 'LCL and UCL', quantsPCapt)


    def plotCentralDensity(self):
        P.figure(figsize=(18,6))
        cc = 1
        for spp in self.allSpp:
            P.subplot(1,3, cc)
            meanCD = np.mean(self.centralDensity[spp], axis = 0)
            quants = mquantiles(self.centralDensity[spp], axis = 0, prob = [0.025, 0.975])

            P.plot(self.years, meanCD, color='k', linewidth=3)
            P.plot(self.years, quants[0], '--k')
            P.plot(self.years, quants[1], '--k')
            P.xlabel('Years', fontsize = 14)
            P.ylabel('Density of ' + spp + '($ha^{-1}$)', fontsize = 14)
            cc += 1
        fname = 'centralDensity_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()



    def find_Den_PropHR(self):
        """
        ## FIND THE PROP-HR RATIO WHEN DENSITY = IMPACT THRESHOLD
        """
        self.ratioAtDIF = {}
        for spp in self.allSpp:
#            print(spp, 'self.params.trRate5', self.params.trRate5[spp])
            self.ratioAtDIF[spp] = {}
            ## GET RATIO FOR FULL AREA
            self.meanN_values = np.array(self.meanN[spp])
            exceedMask = self.meanN_values >= self.params.trRate5[spp]
            if np.any(exceedMask):
                # np.argmax returns the first index of the maximum value
                min_index = np.argmax(exceedMask)  
            else:
                # or handle the case when there is no True value
                min_index = None  
#            self.ratioAtDIF[spp]['fullAreaDenHorizLine'] = self.params.trRate5[spp]
#            self.ratioAtDIF[spp]['fullAreaDenHorizLine'] = self.meanN_values[min_index]
            diffMeanN = np.abs(self.meanN[spp] - self.params.trRate5[spp])
            minDiff = np.min(diffMeanN)
            minMask = diffMeanN == minDiff
#            print(spp, 'ratioThres', self.propertyHR_Ratio[spp][minMask])
#            print(spp, 'Density Thres', self.meanN[spp][minMask])
            self.ratioAtDIF[spp]['meanNRatioThresh'] = self.propertyHR_Ratio[spp][minMask]
###            self.ratioAtDIF[spp]['meanNRatioThresh'] = self.propertyHR_Ratio[spp][min_index]

            ####################################
            ## GET RATIO AT DIF FOR TRAPPED AREA
            self.meanNTrapArea_values = np.array(self.meanNTrapArea[spp])

            exceedMaskTrap = self.meanNTrapArea_values >= self.params.trRate5[spp]
            if np.any(exceedMaskTrap):
                # np.argmax returns the first index of the maximum value
                if spp == 'Rats':
                    min_index = np.argmax(exceedMaskTrap) - 1
                else:
                    min_index = np.argmax(exceedMaskTrap)
#                self.ratioAtDIF[spp]['trapAreaDenHorizLine'] = self.params.trRate5[spp]
###                self.ratioAtDIF[spp]['trapAreaDenHorizLine'] = self.meanNTrapArea_values[min_index]
                self.ratioAtDIF[spp]['trappedNRatioThresh'] = self.propertyHR_Ratio[spp][min_index]
               
            else:
                # or handle the case when there is no True value
                min_index = None
#                self.ratioAtDIF[spp]['trapAreaDenHorizLine'] = self.params.trRate5[spp]
                self.ratioAtDIF[spp]['trappedNRatioThresh'] = np.max(self.propertyHR_Ratio[spp])

#            print(spp, 'min Index mean n trap area', min_index)



###            diffTrapN = np.abs(self.meanNTrapArea[spp] - self.params.trRate5[spp])
###            minDiffTrap = np.min(diffTrapN)
###            minMaskTrap = diffTrapN == minDiffTrap
###            self.ratioAtDIF[spp]['trappedNRatioThresh'] = self.propertyHR_Ratio[spp][minMaskTrap]


    def plotNStorage(self):
#        self.nStorage = np.array(self.simResultsLists['nStorage']) / self.areaHa *100
#        meanN = np.mean(self.nStorage, axis = 0)
#        quants = mquantiles(self.nStorage, axis = 0, prob = [0.025, 0.975])
        ## PLOT PA RATIO
        P.figure(figsize=(18, 6))
        cc = 1
        for spp in self.allSpp:
            P.subplot(1,3, cc)
            P.plot(self.paRatio[spp][1:], self.meanN[spp][1:], color='k', linewidth=3)
            P.fill_between(self.paRatio[spp][1:], self.nQuants[spp][1:, 0], self.nQuants[spp][1:, 1], 
                alpha = 0.2, color = 'k')
            cc += 1
            P.xlabel('Ratio of property perimeter to area', fontsize = 14)
            P.ylabel('Density of ' + spp + ' ($km^{2}$)', fontsize = 14)
        P.tight_layout()
        fname = 'density_paRatio_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()

        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(18,6))
        cc = 1
        for spp in self.allSpp:
            P.subplot(1,3, cc)
            P.plot(self.propertyHR_Ratio[spp], self.meanN[spp], color='k', linewidth=3)
            P.fill_between(self.propertyHR_Ratio[spp], self.nQuants[spp][:, 0], self.nQuants[spp][:, 1], 
                alpha = 0.2, color = 'k')

            print(spp, 'xmax ratioAtDIF', self.ratioAtDIF[spp]['meanNRatioThresh'],
                'ymax .trRate5', self.params.trRate5[spp])


            P.hlines(y = self.params.trRate5[spp], xmax = self.ratioAtDIF[spp]['meanNRatioThresh'],
                xmin = 0, color = 'r', linestyle = 'dashed')
            P.vlines(x = self.ratioAtDIF[spp]['meanNRatioThresh'], ymin = 0,
                ymax = self.params.trRate5[spp], color = 'r', linestyle = 'dashed')

            cc += 1
            P.xlabel('Ratio of property area to HR area', fontsize = 14)
            P.ylabel('Density of ' + spp + ' ($km^{2}$)', fontsize = 14)
        P.tight_layout()
        fname = 'den_Prop_HR_Ratio_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()


    def plotNTrapAreaStorage(self):
        ## PLOT PA RATIO FOR TRAPPED AREA ONLY
        P.figure(figsize=(18, 6))
        cc = 1
        for spp in self.allSpp:
            P.subplot(1,3, cc)
            P.plot(self.paRatio[spp][1:], self.meanNTrapArea[spp][1:], color='k', linewidth=3)
            P.fill_between(self.paRatio[spp][1:], self.quantsNTrapArea[spp][1:, 0], 
                self.quantsNTrapArea[spp][1:, 1], 
                alpha = 0.2, color = 'k')
            cc += 1
            P.xlabel('Ratio of property perimeter to area', fontsize = 14)
            P.ylabel('Trapped area density of ' + spp + ' ($km^{2}$)', fontsize = 14)
        P.tight_layout()
        fname = 'density_paRatio_Trapped_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()

        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(18,6))
        cc = 1
        for spp in self.allSpp:
            P.subplot(1,3, cc)
            P.plot(self.propertyHR_Ratio[spp], self.meanNTrapArea[spp], color='k', linewidth=3)
            P.fill_between(self.propertyHR_Ratio[spp], self.quantsNTrapArea[spp][:, 0], 
                self.quantsNTrapArea[spp][:, 1], 
                alpha = 0.2, color = 'k')
            cc += 1
            P.xlabel('Ratio of property area to HR area', fontsize = 14)
            P.ylabel('Trapped area density of ' + spp + ' ($km^{2}$)', fontsize = 14)
        P.tight_layout()
        fname = 'den_Prop_HR_Ratio_Trapped_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()


    def plotPEradication(self):
#        eradEventSum = np.array(self.simResultsLists['eradEventSum'])
        print('pErad', self.pErad)
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(8,8))
        for spp in self.allSpp:
            P.plot(self.propertyHR_Ratio[spp], self.pErad[spp], label = spp, linewidth=3)
        P.axhline(y = 0.95, color = 'k', linestyle = 'dashed')
        P.legend(loc = 'upper right')
        P.xlabel('Ratio of property area to HR area', fontsize = 14)
        P.ylabel('Probability of eradication', fontsize = 14)
        fname = 'pEradication_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()


    def plotCosts(self):
#        costs = np.array(self.simResultsLists['cost']) / self.areaHa
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(8,8))
        for spp in self.allSpp:
            P.plot(self.propertyHR_Ratio[spp], self.costs[spp], label = spp, linewidth=3)
    
        P.xlabel('Ratio of property area to HR area', fontsize = 16)
        P.ylabel('Annual trapping costs ($ \\$ ha^{-1}$)', fontsize = 16)
        P.legend(loc = 'upper right')
        fname = 'trappingCosts_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()

        ## PLOT PROPERTY PERIMETER IN METRES VS COST
        P.figure(figsize=(8,8))
        for spp in self.allSpp:
            P.plot(self.propCircumference[spp], self.costDenseTrapping[spp], label = spp, linewidth=3)
        P.xlabel('Property circumference (m)', fontsize = 16)
        P.ylabel('Annual trapping costs in buffer area ($)', fontsize = 16)
        P.legend(loc = 'upper left')
        fname = 'trapCosts_Perimeter_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()



    def plotDensityCost(self):
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(18, 6))
        cc = 1
        for spp in self.allSpp:
            P.subplot(1,3, cc)
            ax1 = P.gca()
            ax1.plot(self.propertyHR_Ratio[spp], self.meanN[spp], color='b', 
                label = 'Density of ' + spp, linewidth=3)
            ax1.fill_between(self.propertyHR_Ratio[spp], self.nQuants[spp][:, 0], 
                self.nQuants[spp][:, 1], alpha = 0.2, color = 'b')

            # Calculate relative x and y max
            current_xlim = ax1.get_xlim()
            relative_xmax = (self.ratioAtDIF[spp]['meanNRatioThresh'] - 
                current_xlim[0]) / (current_xlim[1] - current_xlim[0])

            print('relative_xmax', relative_xmax)

            current_ylim = ax1.get_ylim()
            relative_ymax = (self.params.trRate5[spp] - 
                current_ylim[0]) / (current_ylim[1] - current_ylim[0])

            print('relative_ymax', relative_ymax, 'tr rates', self.params.trRate5[spp])


            ax1.axhline(y = self.params.trRate5[spp], 
                xmax = relative_xmax,
                xmin = 0, color = 'r', linestyle = 'dashed')
            ax1.axvline(x = self.ratioAtDIF[spp]['meanNRatioThresh'], ymin = 0,
                ymax = relative_ymax, color = 'r', linestyle = 'dashed')



            ax2 = ax1.twinx()
            ax2.plot(self.propertyHR_Ratio[spp], self.costs[spp], color='k', linewidth=3)
            ax1.set_xlabel('Ratio of property area to HR area', fontsize = 14)
            if cc == 1:
                ax1.set_ylabel('Density ($km^{2}$)', fontsize = 14)
            else:
                ax1.set_ylabel('')
            ax1.spines["right"].set_edgecolor("blue")
            ax1.tick_params(axis='y', colors="blue")
            ax1.yaxis.label.set_color("blue")
            ax1.legend(loc = 'upper right')
            if cc == 3:
                ax2.set_ylabel('Annual trapping costs ($ \\$ ha^{-1}$)', fontsize = 14)
            else:
                ax2.set_ylabel('')
            cc += 1
        P.tight_layout()
        fname = 'den_Cost_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()


    def plotDensityTrapArea(self):
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(18,6))
        cc = 1
        for spp in self.allSpp:
            P.subplot(1,3, cc)
            ax1 = P.gca()
            ax1.plot(self.propertyHR_Ratio[spp], self.meanNTrapArea[spp], color='b', 
                label = 'Density of ' + spp, linewidth=3)
            ax1.fill_between(self.propertyHR_Ratio[spp], self.quantsNTrapArea[spp][:,0], 
                self.quantsNTrapArea[spp][:, 1], alpha = 0.2, color = 'b')

           # Calculate relative x and y max
            current_xlim = ax1.get_xlim()
            relative_xmax = (self.ratioAtDIF[spp]['trappedNRatioThresh'] - 
                current_xlim[0]) / (current_xlim[1] - current_xlim[0])

            current_ylim = ax1.get_ylim()
            relative_ymax = (self.params.trRate5[spp] - 
                current_ylim[0]) / (current_ylim[1] - current_ylim[0])

            print(spp, 'break: xmax', relative_xmax)

            ax1.axhline(y = self.params.trRate5[spp], 
                xmax = relative_xmax,
                xmin = 0, color = 'r', linestyle = 'dashed')
            ax1.axvline(x = self.ratioAtDIF[spp]['trappedNRatioThresh'], ymin = 0,
                ymax = relative_ymax, color = 'r', linestyle = 'dashed')


            ax2 = ax1.twinx()
            ax2.plot(self.propertyHR_Ratio[spp], self.costs[spp], color='k', linewidth=3)
            ax1.set_xlabel('Ratio of property area to HR area', fontsize = 14)
            if cc == 1:
                ax1.set_ylabel('Density in trapped area ($km^{2}$)', fontsize = 14)
            else:
                ax1.set_ylabel('')
            ax1.spines["right"].set_edgecolor("blue")
            ax1.tick_params(axis='y', colors="blue")
            ax1.yaxis.label.set_color("blue")
            ax1.legend(loc = 'upper right')
            if cc == 3:
                ax2.set_ylabel('Annual trapping costs ($ \\$ ha^{-1}$)', fontsize = 14)
            else:
                ax2.set_ylabel('')
            cc += 1
        P.tight_layout()
        fname = 'denTrapArea_Cost_AllSpp.png'
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()
        P.close()


    def plot2RowsDensityCost(self):
        ## PLOT PROPERTY AREA TO HR AREA RATIO
        P.figure(figsize=(15, 10))
        cc = 1
        for spp in self.allSpp:
            P.subplot(2,3, cc)
            ax1 = P.gca()
            ax1.plot(self.propertyHR_Ratio[spp], self.meanN[spp], color='b', 
                label = 'Density of ' + spp, linewidth=3)
            ax1.fill_between(self.propertyHR_Ratio[spp], self.nQuants[spp][:, 0], 
                self.nQuants[spp][:, 1], alpha = 0.2, color = 'b')

            # Calculate relative x and y max
            current_xlim = ax1.get_xlim()
            relative_xmax = (self.ratioAtDIF[spp]['meanNRatioThresh'] - 
                current_xlim[0]) / (current_xlim[1] - current_xlim[0])

            current_ylim = ax1.get_ylim()
#            relative_ymax = (self.ratioAtDIF[spp]['fullAreaDenHorizLine'] - 
#                current_ylim[0]) / (current_ylim[1] - current_ylim[0])
            relative_ymax = (self.params.trRate5[spp] - 
                current_ylim[0]) / (current_ylim[1] - current_ylim[0])

            ax1.axhline(y = self.params.trRate5[spp], 
                xmax = relative_xmax[0],
                xmin = 0, color = 'r', linestyle = 'dashed')
            ax1.axvline(x = self.ratioAtDIF[spp]['meanNRatioThresh'], ymin = 0,
                ymax = relative_ymax, color = 'r', linestyle = 'dashed')


            baseCost = self.costs[spp][0]
            print(spp, 'Base Cost: ', baseCost, 'maxCost', np.max(self.costs[spp]))
            costRatio = self.costs[spp] / baseCost

            ax2 = ax1.twinx()
            ax2.plot(self.propertyHR_Ratio[spp], self.costs[spp], color='k', linewidth=3)
#            ax2.plot(self.propertyHR_Ratio[spp], costRatio, color='k', linewidth=3)
            ax1.set_xlabel('')
            if cc == 1:
                ax1.set_ylabel('Density over entire area ($km^{-2}$)', fontsize = 14)
            else:
                ax1.set_ylabel('')
            ax1.spines["right"].set_edgecolor("blue")
            ax1.tick_params(axis='y', colors="blue")
            ax1.yaxis.label.set_color("blue")
#            ax1.legend(loc = 'upper left')
            if cc == 3:
                ax2.set_ylabel('Annual trapping cost ($ \\$ ha^{-1}$)', fontsize = 14)
#                ax2.set_ylabel('Annual proportional cost increase', fontsize = 14)
            else:
                ax2.set_ylabel('')
            cc += 1



            ax1.text(0.05, 0.97, spp, transform=ax1.transAxes, fontsize=16, 
                verticalalignment='top', color='blue')


        for spp in self.allSpp:
            P.subplot(2,3, cc)
            ax3 = P.gca()
            ax3.plot(self.propertyHR_Ratio[spp], self.meanNTrapArea[spp], color='b', 
                label = spp, linewidth=3)
            ax3.fill_between(self.propertyHR_Ratio[spp], self.quantsNTrapArea[spp][:,0], 
                self.quantsNTrapArea[spp][:, 1], alpha = 0.2, color = 'b')

           # Calculate relative x and y max
            current_xlim = ax3.get_xlim()
            relative_xmax = (self.ratioAtDIF[spp]['trappedNRatioThresh'] - 
                current_xlim[0]) / (current_xlim[1] - current_xlim[0])

            current_ylim = ax3.get_ylim()
            relative_ymax = (self.params.trRate5[spp] - 
                current_ylim[0]) / (current_ylim[1] - current_ylim[0])


            print('############ DEBUG TR DEN', spp, self.params.trRate5[spp])

            ax3.axhline(y = self.params.trRate5[spp], 
                xmax = relative_xmax,
                xmin = 0, color = 'r', linestyle = 'dashed')
            ax3.axvline(x = self.ratioAtDIF[spp]['trappedNRatioThresh'], ymin = 0,
                ymax = relative_ymax, color = 'r', linestyle = 'dashed')


#            ax4 = ax3.twinx()
#            ax4.plot(self.propertyHR_Ratio[spp], self.costs[spp], color='k', linewidth=3)
            ax3.set_xlabel('Ratio of property area to HR area', fontsize = 14)
            if cc == 4:
                ax3.set_ylabel('Density in trapped area ($km^{-2}$)', fontsize = 14)
            else:
                ax3.set_ylabel('')
            ax3.spines["right"].set_edgecolor("blue")
            ax3.tick_params(axis='y', colors="blue")
            ax3.yaxis.label.set_color("blue")
#            ax3.legend(loc = 'upper right')
#            if cc == 3:
#                ax4.set_ylabel('Annual trapping costs ($ \\$ ha^{-1}$)', fontsize = 14)
#            else:
#                ax4.set_ylabel('')
            cc += 1
        P.tight_layout()
        fname = 'den_2Rows_TrapArea_Cost_AllSpp.png'
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
        ax2.set_ylabel('Trapping costs ($\\$ ha^{-1}$)', fontsize = 16)
        fname = 'pErad_Cost_{}.png'.format(self.params.species)
        pathFName = os.path.join(self.params.outputDataPath, fname)
        P.savefig(pathFName, format='png', dpi = 300)
        #P.show()




########            Main function
#######
def main():

    params = ownerParallel.Params()
    processresults = ProcessResults(params)

    maxMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max Mem Usage', maxMem)


if __name__ == '__main__':
    main()
