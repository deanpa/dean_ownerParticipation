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
import resource

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
def loopYears(prop, nProperties, startDensity, areaHa, nStorage, nTrappingArea, extentSide,
        sigma, g0, nYears, trapX, trapY, trapNights, pTrapFail, pNeoPhobic,
        adultSurv, adultSurvDecay, perCapRecruit, recruitDecay, dispersalSD, 
        kSpp, centre, DDRadius, DDArea, eradEventSum, centralDensity, propRadius, pCapture):
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

            if prop == 0 & yr == 0:
                pCapture[ind] = pCaptAll

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
            nDD = np.sum(distDD < DDRadius) / DDArea
            pSurv = adultSurv * (np.exp(-nDD**2 / kSpp**adultSurvDecay))
#            pSurv = adultSurv * (np.exp(-nDD**2 / kDistanceDD**adultSurvDecay))
            pMaxRec = np.exp(-nDD**2 / kSpp**recruitDecay)
#            pMaxRec = np.exp(-nDD**2 / kDistanceDD**recruitDecay)
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
#                            print('OUTSIDE', 'recX', recX, 'recY', recY, 
#                                'extentSide', extentSide)
                        upY = np.min(np.array([Y[indx] + (.33*extentSide), extentSide]))
                        loY = np.max(np.array([Y[indx] - (.33*extentSide), 0]))
                        lx = np.max(np.array([X[indx] - (.33*extentSide), 0]))
                        rx = np.min(np.array([X[indx] + (.33*extentSide), extentSide]))
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
        centralDensity[prop, yr] = densityHA

    nTrapArea = np.sum(distCentre >= propRadius[prop])
#        print('i', i, 'prop', prop, 'nTrapArea', nTrapArea)
    nTrappingArea[prop] = nTrapArea

    nStorage[prop] = N
    if N == 0:
        eradEventSum[prop] += 1

#        print('prop', prop, 'iter', i, centralDensity[prop, i])
#            print('prop central density', centralDensity[iter])
#    print('End centralDensity', centralDensity[0])
    return(X,Y)



class Params(object):
    def __init__(self):
        self.model = 'Model2'
        self.species = 'Stoats'
        self.k = {'Rats' : 5.0, 'Possums' : 8.0, 'Stoats' : 2.2}
        self.sigma = {'Rats' : 40, 'Possums' : 80, 'Stoats' : 300}
        self.g0 = {'Rats' : .05, 'Possums' : 0.1, 'Stoats' : 0.02}

        self.startDensity = {'Rats' : 4, 'Possums' : 7, 'Stoats' : 0.03}
        self.propHrMultiplier = [.5, 4.0]    # 2.0]
        self.extentHRMultiplier = 10
        self.dispersalSD = {'Rats' : 300, 'Possums' : 500, 'Stoats' : 1000}

        ## CABP DOC RECOMMENDATIONS
        self.trapLayout = {'Rats' : {'transectDist' : 100, 'trapDist' : 50}, 
                            'Possums' : {'transectDist' : 200, 'trapDist' : 50},
                            'Stoats' : {'transectDist' : 800, 'trapDist' : 200}}

#        self.trapLayout = {'Rats' : {'transectDist' : 200, 'trapDist' : 50}, 
#                            'Possums' : {'transectDist' : 200, 'trapDist' : 50},
#                            'Stoats' : {'transectDist' : 1000, 'trapDist' : 200}}



        self.bufferLayout = {'Rats' : {'transectDist' : 75, 'trapDist' : 25}, 
                            'Possums' : {'transectDist' : 100, 'trapDist' : 25},
                            'Stoats' : {'transectDist' : 750, 'trapDist' : 100}} 

        self.bufferHRProp = 2.0
        self.adultSurv = {'Rats' : np.exp(-0.79850769621), 'Possums' :  np.exp(-0.25), 
            'Stoats' : np.exp(-0.5)}
        self.adultSurvDecay = {'Rats' : 2.1, 'Possums' : 3.0, 'Stoats' : 2.5}
        self.perCapRecruit = {'Rats' : 4.5, 'Possums' : 0.8, 'Stoats' : 4.5}
        self.recruitDecay = {'Rats' : 1.65, 'Possums' : 1.93, 'Stoats' : 1.5}
        self.distanceDD = {'Rats' : 1.5, 'Possums' : 1.5, 'Stoats' : 1.5}

        ## COST PARAMETERS
        self.costPerTrap = {'Rats' : 5.0, 'Possums' : 5.0, 'Stoats' : 25.0}
        self.nCheckedPerDay = {'Rats' : 100.0, 'Possums' : 100.0, 'Stoats' : 50.0}
        self.dayRate = 400.0
        self.nRecheckPerYear = {'Rats' : 3.0, 'Possums' : 4.0, 'Stoats' : 3.0}
        self.trapNightsPerSet = {'Rats' : 10.0, 'Possums' : 1.0, 'Stoats' : 9.0}

        self.iter = 1
        self.nYears = 8
        self.pTrapFail = 0.02
        self.pNeoPhobic = 0.03

        ## DENSITY PER KM SQUARED RESULTING IN 5% TRACKING RATE
        self.trRate5 = {'Rats' : 20.0, 'Possums' : 13.0, 'Stoats' : 0.4}





        ## GET USER
        userName = getpass.getuser()
        resultsPath = os.path.join('DataResults', 'Results', userName, 'OwnerParticipation',
            self.model)

        baseDir = os.getenv('BROADSCALEDIR', default='.')
        if baseDir == '.':
            baseDir = '/home/dean/workfolder/projects/dean_ownerParticipation/'
            self.outputDataPath = os.path.join(baseDir, resultsPath)
        else:
            # ## ON NESI
            nesiNoBackup = '/nesi/nobackup/landcare04126/'
#            baseDir = os.path.join(baseDir, 'DataResults', 'Results') 
            self.outputDataPath = os.path.join(nesiNoBackup, resultsPath)









#        baseDir = os.getenv('BROADSCALEDIR', default='.')
#        if baseDir == '.':
#            baseDir = '/home/dean/workfolder/projects/dean_ownerParticipation/DataResults/Results/'
#        else:
#            ## IF ON NESI, ADD THE PredatorFree BASE DIRECTORY
#            baseDir = os.path.join(baseDir, 'DataResults', 'Results') 
#        ## GET USER
#        userName = getpass.getuser()
#        resultsPath = os.path.join(userName, 'OwnerParticipation')
#        ## PUT TOGETHER THE BASE DIRECTORY AND PATH TO RESULTS DIRECTORY 
#        self.outputDataPath = os.path.join(baseDir, resultsPath)
        

        self.jobID = int(os.getenv('SLURM_ARRAY_TASK_ID', default = '0'))
        print('jobID', self.jobID)
        print('Results directory:', self.outputDataPath)
        print('############################')
        ## MAKE NEW RESULTS DIRECTORY IF DOESN'T EXIST
        if not os.path.isdir(self.outputDataPath):
            os.makedirs(self.outputDataPath)
        fName = 'ownerResults_{}_Job_{}.json'.format(self.species, self.jobID)
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
        self.nStorage = np.zeros(self.nProperties)
        self.nTrappingArea = np.zeros(self.nProperties)
        self.eradEventSum = np.zeros(self.nProperties)
        self.centralDensity = np.zeros((self.nProperties, self.params.nYears))
        self.nTrapsStorage = []     #np.zeros(self.nProperties)
        self.pCapture = np.zeros(int(self.params.startDensity[self.params.species] * self.areaHa * 2))

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

            (X,Y) = loopYears(prop, self.nProperties, 
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
                self.params.k[self.params.species], self.centre, self.DDRadius, self.DDArea,
                self.eradEventSum, self.centralDensity, self.propRadius, self.pCapture)

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
        simResults['pCapture'] = self.pCapture.tolist()
        ## WRITE RESULTS TO JSON
        with open(self.params.simResultsPath, 'w') as f:
            json.dump(simResults, f)



########            Main function
#######
def main():
    #foo2(2,12,10)

    params = Params()
    simulation = Simulation(params)

    maxMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max Mem Usage', maxMem)


if __name__ == '__main__':
    main()
