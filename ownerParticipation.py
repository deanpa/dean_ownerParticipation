#!/usr/bin/env python

import os
from pathlib import Path
import getpass
import numpy as np
from scipy import stats
import pylab as P
from numba import njit
from scipy.stats.mstats import mquantiles
import json
import csv

@njit
def distFX(x1, y1, x2, y2):
    """
    calc distance matrix between 2 sets of points
    resulting matrix dimensions = (len(x1), len(x2))
    """
    deltaX_sq = (x1 - x2)**2
    deltaY_sq = (y1 - y2)**2
    dist = np.sqrt(deltaX_sq + deltaY_sq)
    return dist

@njit
def foo2(l,h,n):
    ll = list(np.random.uniform(l, h, n))
    for i in range(3):
        newList = list(np.random.uniform(l, h, n))
        ll.extend(newList)
        m = np.min(np.array([6, (3+ll[-1])]))
        print('m', m)
    print('list', ll)
    del(ll[3])
    return(ll)
#    return(newList)

@njit
def loopIterations(iter, prop, nProperties, startDensity, areaHa, nStorage, nTrappingArea, extentSide,
        sigma, g0, nYears, trapX, trapY, trapNights, pTrapFail, pNeoPhobic,
        adultSurv, adultSurvDecay, perCapRecruit, recruitDecay, dispersalSD, 
        kDistanceDD, centre, DDRadius, DDArea, eradEventSum, centralDensity, propRadius):
    for i in range(iter):
#        N = 500
        N = np.random.poisson(startDensity * areaHa)
        X = list(np.random.uniform(0, extentSide, N))
        Y = list(np.random.uniform(0, extentSide, N))
        for yr in range(nYears):
            recruitX = []
            recruitY = []
            nRemoved = 0
            for ind in range(N):
                indx = ind - nRemoved

#                print('ind', ind, 'nRem', nRemoved, 'INDX', indx, 'N', len(X), 'X', X[indx])

                dist = distFX(X[indx], Y[indx], trapX, trapY)
                pCapt = g0 * np.exp(-(dist**2 / 2.0 / sigma**2))
                pNoTrap = 1 - (pCapt * (1.0 - pTrapFail))
                pNoTrap = pNoTrap**trapNights
                pCaptAll = 1.0 - (np.prod(pNoTrap) * (1.0 - pNeoPhobic))
                capt_ind = np.random.binomial(1, pCaptAll)
                if capt_ind == 1:
                    del(X[indx])
                    del(Y[indx])
                    nRemoved += 1
                    continue
            N = len(X)
            ## LOOP INDIVIDUALS AGAIN FOR SURVIVAL, RECRUITMENT AND DISPERSAL                    
            nRemoved = 0
#            print('prop', prop, 'iter', i, 'yr', yr , 'N FOLLOWING TRAPPING', N)
            for ind in range(N):
                indx = ind - nRemoved

#                print('ind', ind, 'nRem', nRemoved, 'INDX', indx, 'N', len(X), 'X', X[indx])

                ## GET LOCAL DD
                distDD = distFX(X[indx], Y[indx], np.array(X), np.array(Y))
                nDD = np.sum(distDD < DDRadius)
                pSurv = adultSurv * (np.exp(-nDD**2 / kDistanceDD**adultSurvDecay))
                pMaxRec = np.exp(-nDD**2 / kDistanceDD**recruitDecay)
                recRate = perCapRecruit * pMaxRec

                surv_ind = np.random.binomial(1, pSurv)
                if surv_ind == 0:
                    del(X[indx])
                    del(Y[indx])
                    nRemoved += 1
                else:
                    nRecruit = np.random.poisson(recRate)
#                    print('ind survived', ind, 'Indx', indx, 'nRecruit', nRecruit, 'recRate', recRate,
#                        'X', X[indx])
                    for rec in range(nRecruit):
                        dx = np.random.normal(0, dispersalSD)
                        dy = np.random.normal(0, dispersalSD)
                        recX = X[indx] + dx
                        recY = Y[indx] + dy
                        xOut = recX < 0 or recX > extentSide
                        yOut = recY < 0 or recY > extentSide
#                        print('rec', rec, 'recX', recX)
                        if xOut or yOut:
#                            print('OUTSIDE', 'recX', recX, 'recY', recY, 'extentSide', extentSide)
                            upY = np.min(np.array([Y[indx] + 100, extentSide]))
                            loY = np.max(np.array([Y[indx] - 100, 0]))
                            lx = np.max(np.array([X[indx] - 100, 0]))
                            rx = np.min(np.array([X[indx] + 100, extentSide]))
                            x = np.random.uniform(rx, lx)
                            y = np.random.uniform(loY, upY)
                            recX = x
                            recY = y
#                            print('rec', rec, 'FIXED LOC', 'recX', recX, 'recY', recY, 
#                                'X', X[indx], 'Y', Y[indx])

                        recruitX.append(recX)
                        recruitY.append(recY)
            X.extend(recruitX)
            Y.extend(recruitY)    
            N = len(X)
#            print('i', i, 'yr', yr, 'prop', prop, 'N', N, 'n Rec', len(recruitX))    

#            if prop == 8:
                ## PLOT CENTRAL DENSITY FOR LAST PROPERTY
            distCentre = distFX(centre[0], centre[1], np.array(X), np.array(Y))
            nDD = np.sum(distCentre < DDRadius)
            densityHA = nDD / DDArea
            centralDensity[prop, i, yr] = densityHA

        nTrapArea = np.sum(distCentre >= propRadius[prop])
#        print('i', i, 'prop', prop, 'nTrapArea', nTrapArea)
        nTrappingArea[i, prop] = nTrapArea

        nStorage[i, prop] = N
        if N == 0:
            eradEventSum[prop] += 1

#        print('prop', prop, 'iter', i, centralDensity[prop, i])
#            print('prop central density', centralDensity[iter])
#    print('End centralDensity', centralDensity[0])
    return(X,Y)



class Params(object):
    def __init__(self):
        self.species = 'Stoats'
#        self.r = {'Rats' : 3.0, 'Possums' : np.log(.75), 'Stoats' : np.log(3.0)}
        self.k = {'Rats' : 4.0, 'Possums' : 8.0, 'Stoats' : 3.5}
        self.sigma = {'Rats' : 40, 'Possums' : 90, 'Stoats' : 300}
        self.g0 = {'Rats' : .05, 'Possums' : 0.1, 'Stoats' : 0.02}

        self.startDensity = {'Rats' : 3, 'Possums' : 7, 'Stoats' : 3}
        self.propHrMultiplier = [.5, 7.57]     # 2.0]
        self.extentHRMultiplier = 10
        self.dispersalSD = {'Rats' : 80, 'Possums' : 150, 'Stoats' : 500}
        self.trapLayout = {'Rats' : {'transectDist' : 100, 'trapDist' : 50}, 
                            'Possums' : {'transectDist' : 100, 'trapDist' : 50},
                            'Stoats' : {'transectDist' : 1000, 'trapDist' : 200}}
        self.bufferLayout = {'Rats' : {'transectDist' : 75, 'trapDist' : 25}, 
                            'Possums' : {'transectDist' : 75, 'trapDist' : 25},
                            'Stoats' : {'transectDist' : 750, 'trapDist' : 100}}
        self.bufferHRProp = 2.0
        self.adultSurv = {'Rats' : np.exp(-0.5), 'Possums' :  np.exp(-0.3), 
            'Stoats' : np.exp(-0.5)}
        self.adultSurvDecay = {'Rats' : 2.6, 'Possums' : 2.0, 'Stoats' : 1.6}
        self.perCapRecruit = {'Rats' : 3.0, 'Possums' : 0.75, 'Stoats' : 3.5}
        self.recruitDecay = {'Rats' : 1.75, 'Possums' : 1.5, 'Stoats' : 1.4}
        self.distanceDD = {'Rats' : 1, 'Possums' : 2, 'Stoats' : 1}

        ## COST PARAMETERS
        self.costPerTrap = {'Rats' : 5.0, 'Possums' : 25.0, 'Stoats' : 25.0}
        self.nCheckedPerDay = {'Rats' : 100.0, 'Possums' : 40.0, 'Stoats' : 50.0}
        self.dayRate = 400.0
        self.nRecheckPerYear = {'Rats' : 3.0, 'Possums' : 6.0, 'Stoats' : 3.0}
        self.trapNightsPerSet = {'Rats' : 10.0, 'Possums' : 1.0, 'Stoats' : 9.0}

        self.iter = 1
        self.nYears = 2
        self.pTrapFail = 0.02
        self.pNeoPhobic = 0.03

        print('Species', self.species, 'Iterations', self.iter, 'Years', self.nYears)

        baseDir = os.getenv('BROADSCALEDIR', default='.')
        if baseDir == '.':
            baseDir = '/home/dean/workfolder/projects/dean_predatorfree/DataResults/Results/'
        else:
            ## IF ON NESI, ADD THE PredatorFree BASE DIRECTORY
            baseDir = os.path.join(baseDir, 'PredatorFree', 'DataResults', 'Results') 
        ## GET USER
        userName = getpass.getuser()
        resultsPath = os.path.join(userName, 'OwnerParticipation')
        ## PUT TOGETHER THE BASE DIRECTORY AND PATH TO RESULTS DIRECTORY 
        self.outputDataPath = os.path.join(baseDir, resultsPath)

        print('Results directory:', self.outputDataPath)
        print('############################')
        ## MAKE NEW RESULTS DIRECTORY IF DOESN'T EXIST
        if not os.path.isdir(self.outputDataPath):
            os.makedirs(self.outputDataPath)
        fName = 'ownerResults_{}.json'.format(self.species)
        self.simResultsPath = os.path.join(self.outputDataPath, fName)

class Simulation(object):
    def __init__(self, params):
        self.params = params
        ####################
        ## RUN FUNCTIONS
        self.getElements()
        self.makeTrapNetworks()
        self.loopProperties()
#        self.plotCentralDensity()
        self.prepareWriteJSON()
        ####################

    def getElements(self):
        self.hrRadius = int(self.params.sigma[self.params.species] * 3)
        self.hrDiameter = self.hrRadius * 2
        print('hrDiameter', self.hrDiameter, 'hr area (ha)', np.pi * self.hrRadius**2 / 10000)
        self.extentSide = self.hrDiameter * self.params.extentHRMultiplier
        self.centre = np.array([self.extentSide / 2, self.extentSide / 2])
        self.propRadius = np.arange(self.hrRadius * self.params.propHrMultiplier[0],
            self.hrRadius * self.params.propHrMultiplier[1], 
            self.params.trapLayout[self.params.species]['trapDist'] * .5)
        self.propRadius = np.append(0, self.propRadius)
 
#            self.params.bufferLayout[self.params.species]['trapDist'] * 1.0)
        self.areaHa = self.extentSide**2 / 10000
        self.propAreaHA = np.pi * self.propRadius**2 / 10000
        self.nProperties = len(self.propAreaHA)
        print('n props', self.nProperties, 'Extent area Ha', self.areaHa, 
            'extent side', self.extentSide, 'propRadius', self.propRadius)
        self.bufferRadius = ((self.hrRadius * self.params.bufferHRProp) + self.propRadius)
        self.bufferRadius = np.append(0, self.bufferRadius)
        print('Buffer Radius', self.bufferRadius)
        ## CARRYING CAPACITY
#        self.kExtent = self.params.k[self.params.species] * self.areaHa
        self.DDRadius = self.hrRadius * self.params.distanceDD[self.params.species]
        if self.params.species == 'Stoats':
            self.DDArea = np.pi * (self.DDRadius**2) / 1e6
            self.kDistanceDD = self.params.k[self.params.species] * self.DDArea
        else:
            self.DDArea = np.pi * (self.DDRadius**2) / 10000
            self.kDistanceDD = self.params.k[self.params.species] * self.DDArea
        print('DDArea', self.DDArea, 'kdistancedd', self.kDistanceDD)
        ## TRAP NIGHTS        
        self.trapNights = (self.params.nRecheckPerYear[self.params.species] * 
            self.params.trapNightsPerSet[self.params.species])


    def makeTrapNetworks(self):
        xMin = self.params.trapLayout[self.params.species]['transectDist'] / 2
        yMin = self.params.trapLayout[self.params.species]['trapDist'] / 2
        x0 = np.arange(xMin, self.extentSide, 
            self.params.trapLayout[self.params.species]['transectDist'])  
        nX = len(x0)
        y0 = np.arange(yMin, self.extentSide,
            self.params.trapLayout[self.params.species]['trapDist'])
        nY = len(y0)
        self.trapX = np.repeat(x0, nY)
        self.trapY = np.tile(y0, nX)

        ## COST OF FULL PARTICIPATION TRAPPING
        self.nTrapsFullParticipation = len(self.trapX)
        fixedCostTotal_Full = (self.nTrapsFullParticipation * 
            self.params.costPerTrap[self.params.species])
        nDays_Full = (self.nTrapsFullParticipation / 
            self.params.nCheckedPerDay[self.params.species] *
            self.params.nRecheckPerYear[self.params.species])
        labourCost_Full = nDays_Full * self.params.dayRate
        self.totalCost_Full = fixedCostTotal_Full + labourCost_Full
        
        xMin = self.params.bufferLayout[self.params.species]['transectDist'] / 2
        yMin = self.params.bufferLayout[self.params.species]['trapDist'] / 2
        x0 = np.arange(xMin, self.extentSide, 
            self.params.bufferLayout[self.params.species]['transectDist'])  
        nX = len(x0)
        y0 = np.arange(yMin, self.extentSide,
            self.params.bufferLayout[self.params.species]['trapDist'])
        nY = len(y0)
        self.bufferX = np.repeat(x0, nY)
        self.bufferY = np.tile(y0, nX)

        self.distCentreTrap = distFX(self.centre[0], self.centre[1], 
            self.trapX, self.trapY)
        self.distbuffTrapCentre = distFX(self.centre[0], self.centre[1], 
            self.bufferX, self.bufferY)
        ## MAKE STORAGE ARRAYS
        self.costStorage = np.zeros(self.nProperties)
        self.costDenseTrapping = np.zeros(self.nProperties)
        self.nStorage = np.zeros((self.params.iter, self.nProperties))
        self.nTrappingArea = np.zeros((self.params.iter, self.nProperties))
        self.eradEventSum = np.zeros(self.nProperties)
        self.centralDensity = np.zeros((self.nProperties, self.params.iter, 
            self.params.nYears))
        self.nTrapsStorage = []     #np.zeros(self.nProperties)

    def plotCentralDensity(self):
        P.figure(figsize=(9,11))
        meanDen = np.mean(self.centralDensity[-1], axis = 0)
        years = np.arange(self.params.nYears)
        P.plot(years, meanDen, 'k-o')
        P.show()


    def loopProperties(self):
#        for prop in range(7,9):
        for prop in range(self.nProperties):
#            print('prop', prop, 'prop radius', self.propRadius[prop], 
#                'buf rad', self.bufferRadius[prop])

            if prop == 0:
                xTrap_prop = self.trapX.copy()
                yTrap_prop = self.trapY.copy()
                self.nTraps_prop = len(self.trapX)
                self.nTrapsStorage.append(self.nTraps_prop) 
                ## CALCULATE COSTS
                self.costStorage[prop] = self.totalCost_Full

            else:
                ## GET TRAP DATA FOR THIS PROPERTY SIZE
                minBuffMask = self.distbuffTrapCentre >= self.propRadius[prop]
                maxBuffMask = self.distbuffTrapCentre <= self.bufferRadius[prop]
                buffMask = minBuffMask & maxBuffMask
                xTrap_buf = self.bufferX[buffMask]
                yTrap_buf = self.bufferY[buffMask]
                mask_trap = self.distCentreTrap > self.bufferRadius[prop]

                trapMinBufMask = self.distCentreTrap >= self.propRadius[prop]
                trapMaxBufMask = self.distCentreTrap <= self.bufferRadius[prop]
                trapBuffMask = trapMinBufMask & trapMaxBufMask

                xTrap_prop = np.append(xTrap_buf, self.trapX[mask_trap])
                yTrap_prop = np.append(yTrap_buf, self.trapY[mask_trap])
                self.nTraps_prop = len(xTrap_prop)
                self.nTrapsStorage.append(self.nTraps_prop) 

                ## CALCULATE COSTS
                self.n_denseTrap_buf = np.sum(buffMask) 
                self.n_normTrap_buf = np.sum(trapBuffMask)
                self.calcCost(prop)

            (X,Y) = loopIterations(self.params.iter, prop, self.nProperties, 
                self.params.startDensity[self.params.species], self.areaHa, 
                self.nStorage, self.nTrappingArea, self.extentSide, 
                self.params.sigma[self.params.species], 
                self.params.g0[self.params.species], self.params.nYears,
                xTrap_prop, yTrap_prop, self.trapNights, self.params.pTrapFail,
                self.params.pNeoPhobic, self.params.adultSurv[self.params.species],
                self.params.adultSurvDecay[self.params.species], 
                self.params.perCapRecruit[self.params.species], 
                self.params.recruitDecay[self.params.species], 
                self.params.dispersalSD[self.params.species],
                self.kDistanceDD, self.centre, self.DDRadius, self.DDArea,
                self.eradEventSum, self.centralDensity, self.propRadius)

        ## DEBUGGING
        trapMinBufMask = self.distCentreTrap >= self.propRadius[prop]
        trapMaxBufMask = self.distCentreTrap <= self.bufferRadius[prop]
        trapBuffMask = trapMinBufMask & trapMaxBufMask
        trapNormalMask = self.distCentreTrap > self.bufferRadius[prop]
        ############

        ## DEBUGGING
#        debugXBuf = xTrap_buf
#        debugYBuf = yTrap_buf
        xydataBuf = np.zeros((len(xTrap_buf), 2))
        xydataBuf[:,0] = xTrap_buf
        xydataBuf[:,1] = yTrap_buf

        self.writeCSV('Buf', xydataBuf)

        trapDebugX = self.trapX[trapNormalMask]
        trapDebugY = self.trapY[trapNormalMask]
#        trapDebugX = self.trapX[trapBuffMask]
#        trapDebugY = self.trapY[trapBuffMask]
        xydataTrap = np.zeros((len(trapDebugX),2))
        xydataTrap[:,0] = trapDebugX
        xydataTrap[:,1] = trapDebugY
        self.writeCSV('TrapNormal', xydataTrap)

    def writeCSV(self, n, xy):
        fname = 'xyDat{}.csv'.format(n)
        # Write the data to the CSV file
        with open(fname, 'w', newline='') as csvfile:
            # Create a CSV writer object
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(['X', 'Y'])
            # Write each row of data
            writer.writerows(xy)


    def calcCost(self, prop):
        ## CALCULATE AREA OF BUFFER
        propArea = np.pi * self.propRadius[prop]**2
        buffAreaHA = ((np.pi * self.bufferRadius[prop]**2) - propArea) / 10000

        ## COST OF FULL PARTICIPATION TRAPPING
        fixedCostTotal_norm = (self.n_normTrap_buf * 
            self.params.costPerTrap[self.params.species])
        nDays_norm = (self.n_normTrap_buf / 
            self.params.nCheckedPerDay[self.params.species] *
            self.params.nRecheckPerYear[self.params.species])
        labourCost_norm = nDays_norm * self.params.dayRate
        totalCost_norm = fixedCostTotal_norm + labourCost_norm
        
        ## NORMAL TRAP DENSITY COSTS IN BUFFER
        fixedCostTotal_norm = (self.n_normTrap_buf * 
            self.params.costPerTrap[self.params.species])
        nDays_norm = (self.n_normTrap_buf / 
            self.params.nCheckedPerDay[self.params.species] *
            self.params.nRecheckPerYear[self.params.species])
        labourCost_norm = nDays_norm * self.params.dayRate
        totalCost_norm = fixedCostTotal_norm + labourCost_norm

        ## INCREASED TRAP DENSITY COSTS IN BUFFER
        fixedCostTotal_dense = (self.n_denseTrap_buf * 
            self.params.costPerTrap[self.params.species])
        nDays_dense = (self.n_denseTrap_buf / 
            self.params.nCheckedPerDay[self.params.species] *
            self.params.nRecheckPerYear[self.params.species])
        labourCost_dense = nDays_dense * self.params.dayRate
        totalCost_dense = fixedCostTotal_dense + labourCost_dense
        self.costDenseTrapping[prop] = totalCost_dense

        ## TOTAL COST FROM ALL TRAPS
        fixedCostTotal_All = (self.nTraps_prop * 
            self.params.costPerTrap[self.params.species])
        nDays_All = (self.nTraps_prop / 
            self.params.nCheckedPerDay[self.params.species] *
            self.params.nRecheckPerYear[self.params.species])
        labourCost_All = nDays_All * self.params.dayRate
        totalCost_All = fixedCostTotal_All + labourCost_All
        self.costStorage[prop] = totalCost_All 

        ## INCREASE IN COST FROM ADDING TRAPS IN BUFFER
#        self.costStorage[prop] = (totalCost_dense - totalCost_norm) / 
#        print('n days per year', nDays, 'fixed', fixedCostTotal, 'labour', labourCost,
#        print('diff Cost per ha', self.costStorage[prop], 'n norm', self.n_normTrap_buf,
#            'norm cost', totalCost_norm, 'n dense', self.n_denseTrap_buf,
#            'dense cost', totalCost_dense, 'totalCost All', totalCost_All)



    def prepareWriteJSON(self):
        simResults = {}
        simResults['CentralDensity'] = self.centralDensity.tolist()
        simResults['hrRadius'] = self.hrRadius
        simResults['propRadius'] = self.propRadius.tolist()
        simResults['propArea'] = self.propAreaHA.tolist()
        simResults['nTraps'] = self.nTrapsStorage
        simResults['nStorage'] = self.nStorage.tolist()
        simResults['nTrappingArea'] = self.nTrappingArea.tolist()
        simResults['eradEventSum'] = self.eradEventSum.tolist()
        simResults['cost'] = self.costStorage.tolist()
        simResults['costDenseTrap'] = self.costDenseTrapping.tolist()
        ## WRITE RESULTS TO JSON
        with open(self.params.simResultsPath, 'w') as f:
            json.dump(simResults, f)


class ProcessResults(object):
    def __init__(self, params):
        self.params = params
        ####################
        ## RUN FUNCTIONS
        self.readJSON()
        self.plotCentralDensity()
        self.plotNStorage()
        self.plotPEradication()
        self.plotCosts()
        self.plotDensityCost()
        self.plotDensityTrapArea()
        self.plotPEradCost()


    def readJSON(self):
        # Load the JSON file back into a list
        with open(self.params.simResultsPath, 'r') as json_file:
            self.simResultsLists = json.load(json_file)
        # Convert the list back to a NumPy array
        self.years = np.arange(self.params.nYears)
        self.hrRadius = int(self.params.sigma[self.params.species] * 3)
        self.hrDiameter = self.hrRadius * 2
        self.extentSide = self.hrDiameter * self.params.extentHRMultiplier
        self.areaHa = self.extentSide**2 / 10000

        self.propRadius = np.array(self.simResultsLists['propRadius'])
        self.propCircumference = np.pi * 2.0 * self.propRadius
        self.propArea = np.array(self.simResultsLists['propArea'])
        self.paRatio = np.zeros(len(self.propArea))
        self.paRatio[1:] = self.propCircumference[1:] / self.propArea[1:]  #(m / ha)
#        print('paRatio', self.paRatio, 'area', self.propArea, 
#            'circum', self.propCircumference, 'prop radius', self.propRadius)

        self.hrAreaHa = np.pi * (self.hrRadius**2) / 10000
        self.propertyHR_Ratio = self.propArea / self.hrAreaHa

        self.nStorage = np.array(self.simResultsLists['nStorage']) / self.areaHa *100
        self.meanN = np.mean(self.nStorage, axis = 0)
        self.nQuants = mquantiles(self.nStorage, axis = 0, prob = [0.025, 0.975])
        self.costs = np.array(self.simResultsLists['cost']) / self.areaHa
        self.costDenseTrapping = np.array(self.simResultsLists['costDenseTrap'])

        self.eradEventSum = np.array(self.simResultsLists['eradEventSum'])
        self.pErad = self.eradEventSum / self.params.iter
        print('Prop area to HR area', self.propertyHR_Ratio, 'paRatio', self.paRatio,
            'Prop Area', self.propArea)
        ## N PREDS IN TRAPPING AREA AT END OF OPERATION
        self.areaOutsideProp = self.areaHa - self.propArea
        self.nTrappingArea = np.array(self.simResultsLists['nTrappingArea'])
        densityTrapArea = self.nTrappingArea / self.areaOutsideProp * 100
        self.meanNTrapArea = np.mean(densityTrapArea, axis = 0)
        self.quantsNTrapArea = mquantiles(densityTrapArea, axis = 0, prob = [0.025, 0.975])



    def plotCentralDensity(self):
        centralDensity = np.array(self.simResultsLists['CentralDensity'])[-1]
        meanCD = np.mean(centralDensity, axis = 0)
        quants = mquantiles(centralDensity, axis = 0, prob = [0.025, 0.975])
#        print('cent density shp', centralDensity.shape, 'mean', meanCD, quants)
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
        P.fill_between(self.paRatio[1:], self.nQuants[0,1:], self.nQuants[1, 1:], 
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
        P.fill_between(self.propertyHR_Ratio, self.nQuants[0], self.nQuants[1], 
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
        ax1.fill_between(self.propertyHR_Ratio, self.nQuants[0], self.nQuants[1], 
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
        ax1.fill_between(self.propertyHR_Ratio, self.quantsNTrapArea[0], self.quantsNTrapArea[1], 
            alpha = 0.2, color = 'b', lw = 0)
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

    params = Params()
    simulation = Simulation(params)
    processresults = ProcessResults(params)

    maxMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max Mem Usage', maxMem)


if __name__ == '__main__':
    main()
