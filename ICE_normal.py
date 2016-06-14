import cv2
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import time
from MSE_SSIM import mse
from skimage.measure import structural_similarity as ssim
from small_histogram import hist
from small_histogram_fast import hist as hist2
from rms import rms
import ccv2
import glob

start_time = time.time()

def matchNumNonORB(LoTitles, RefImName,LoImg, RefImg,Method):
	""" Inputs: a list of strings of titles of images, LoTitles
				a string of a reference image RefImName,
				a string representing a method, Method,


		Returns the 'winner', given as the best match between all of the images in LoIm and the single image RefIm"""
	#print "Comparing to", RefIm, "using method", Method
	LoMatches = [[matchFindNonORB(LoTitles[x], RefImName,LoImg[x],RefImg, Method),LoTitles[x]] for x in xrange(len(LoTitles))]
	LoMatches.sort() #lowest print
	if len(LoMatches) <2:
		print "Matches is much too short...something has gone wrong while using", Method
	winner = LoMatches[1][1] #usually, we'll want the first element in the list that is not the original image
	if Method == 'ssim':
		winner = LoMatches[len(LoMatches)-2][1] #for this method, we'll want the biggest number
	#print "The winner is", winner, "using", Method
	return winner #returns name of "best" image. 

#matchNumORB(LoTitles, RefPic,LoImg, RefImg,Method, LoMats, LoDists, LoKP, refKP, NumDist)
def matchNumORB(LoTitles, RefImName,Method, LoMats, LoDists, LoKP, refKP,NumDist  = 0):
	"""		Inputs: a list of strings of titles of images, LoTitles
				a string of a reference image RefImName,
				a string representing a method, Method,



				an optional parameter: the number of distances, NumDist, which will perform the operations on a subset of the information provided
						Any number greater  than 0 for NumDist will truncate the distances """
	#print RefImg
	
	LoMatches = [[matchFindORB(LoTitles[x], RefImName,Method,LoMats[x],LoDists[x],LoKP[x],refKP),LoTitles[x]] for x in xrange(len(LoTitles))]
	LoMatches.sort() #lowest print
	#print LoMatches, Method
	winner = LoMatches[1][1] #usually, we'll want the first element in the list that is not the original image
	if Method == 'homography':
		winner = LoMatches[len(LoMatches)-2][1] #for this method, we'll want the biggest number
	#print "The winner is", winner, "using", Method
	return winner

def matchFindNonORB(TestImName, RefImName,TestIm,RefIm, Method):
	""" Inputs:  two strings: an image name, TestImName, and the name of a reference image, RefImName,
				  a string containing the specified Method used to compare the images
				  an optional parameter: the number of distances, NumDist, which will perform the operations on a subset of the information provided
						Any number greater  than 0 for NumDist will truncate the distances
		Outputs: a number, answer, that comes from the comparison method used between the two images """

	if TestImName == RefImName: #if an image is tested against itself, let's save time by returning a value eliminating it from contention...
		if Method == 'ssim': #for these methods, higher is better, so we return a huge value
			return 1000000
		else: #otherwise, return -1, a number lower than any value we should see...
			return -1

	answer = 0	
	"""Now, the user chooses the method, and the answer gets returned accordingly """	
	if Method == 'mse': #mean squared error
		answer = mse(TestIm,RefIm)
	elif Method == 'ssim': #structural similarity index
		answer = ssim(TestIm,RefIm)
	elif Method == 'rms': #root means squared difference between pixels
		answer = rms(TestIm,RefIm)
	elif Method == 'ccv2':
		answer = ccv2.main(TestImName,RefImName)


	#print TestIm, answer 
	return answer

def matchFindORB(TestImName,RefImName,Method,
				Matches,Distances, kpList1,kpList2):
	""" version of MatchFind that works for the ORB methods"""

	if TestImName == RefImName: #if an image is tested against itself, let's save time by returning a value eliminating it from contention...
		if Method == 'homography': #for this method, higher is better, so we return a huge value
			return 1000000
		else: #otherwise, return -1, a number lower than any value we should see...
			return -1

	if NumDist > 0: #should the user want to use a subset of this information, we will truncate it accordingly...
		Distances = Distances[:NumDist]
		Matches = Matches[:NumDist]

	answer = 0	
	"""Now, the user chooses the method, and the answer gets returned accordingly """	
	if Method == 'median': #median of distances
		answer = np.median(Distances)
	elif Method == 'mean': #mean of distances
		answer = np.mean(Distances)
	elif Method == 'totaldist': #sum up all distances
		answer = sum(Distances)
	elif Method == 'halfdist': #sum up half the distances
		answer = sum(Distances[:len(Distances)/2]) 
	elif Method == 'homography': #homography
		answer = homography(Matches,kpList1,kpList2)

	#print TestIm, answer 
	return answer


def tester(LoTitles,LoMeth,NumDist = 0):
	"""Inputs: a list of strings of titles of images, LoTitles
			   a list of strings representing methods, LoMeth
			   an optional parameter: the number of distances, which will perform the operations on a subset of the information provided
						Any number greater  than 0 for NumDist will truncate the distances

	Tests each image in a list of images, LoIm, against every other image in the list. 

	Prints the 'best' image, as determined by every single method listed in the list of methods, LoM. 
	The 'best' image is the one that appears the most often using all of the methods. """
	for x in xrange(len(LoTitles)): #uses every image as the reference image
		#print "reference:", LoTitles[x] 
		result = singlePicTest(LoTitles,LoTitles[x],LoMeth,NumDist)

		#print "The most likely winner for", LoTitles[x], "is:", result #, "with the list being", LoW


def singleMethTest(LoTitles,Method,NumDist = 0):
	"""Inputs: a list of strings of titles of images, LoTitles
			   a string representing a single method, Method
			   an optional parameter: the number of distances, which will perform the operations on a subset of the information provided
						Any number greater  than 0 for NumDist will truncate the distances

	Tests each image in a list of images, LoIm, against every other image in the list. 

	Prints the 'best' image, as determined by every single method listed in the list of methods, LoM. 
	The 'best' image is the one that appears the most often using Method. """
	#LoTitles,LoIm = imGenerator(LoTitles) #takes the titles, makes images

	for x in xrange(len(LoTitles)): #uses every image as the reference image
		#print "reference:", LoTitles[x] 
			#print #for readability
		result = 0 #initialize result
		if Method in ['histCorr','histChi','histInter','histHell']: #if we want to use one of the histogram methods, we have a separate function for this
			result = hist(LoTitles,LoTitles[x],histDecipher(Method))
			print "The winner is", result[1][1], "using", LoMeth[y]
		else:
			result = matchNum(LoTitles,LoTitles[x],Method,NumDist) #gets the winner for all other methods
		print "The method", Method, "gives a winner of", result, "for", LoTitles[x]

def singlePicTest(LoTitles,RefPic,LoMeth,NumDist = 0):
	"""Inputs: a list of strings of titles of images, LoTitles
			   a string with the title of an image to test against, RefPic
			   a list of strings representing methods, LoMeth
			   an optional parameter: the number of distances, which will perform the operations on a subset of the information provided
						Any number greater  than 0 for NumDist will truncate the distances

	Tests each image in a list of images, LoIm, against every other image in the list. 

	Prints the 'best' image, as determined by every single method listed in the list of methods, LoM. 
	The 'best' image is the one that appears the most often using all of the methods. """
	LoImg = imGenerator(LoTitles) #takes the titles, makes images
	RefImg = cv2.imread(RefPic,0)
	LoW = [] #List of winners...
	ORBneeded = False 
	
	LoKP = [] #initialize all the lists that will hold the ORB information, should we need it
	LoDes = []
	refKP = []
	refDes = []
	LoMats = []
	LoDists = []

	for Algorithm in LoMeth: #let's check if we'll even need to compute ORB, based on the methods in our list...
		if Algorithm in ['median','mean','totaldist','halfdist','homography']: #if one of these methods is in the list, we'll need ORB, so we change our Boolean to True
			ORBneeded = True
			break

	if ORBneeded: #ORB is needed, so let's get that information...
		LoKP,LoDes,refKP,refDes = orbKPGetter(LoImg,RefImg)
		LoMats, LoDists = orbMatchGetter(LoDes,refDes)


	for y in xrange(len(LoMeth)): #checks every method...
		#print #for readability
		Method = LoMeth[y]
		if Method in ['histCorr','histChi','histInter','histHell']: #if we want to use one of the histogram methods, we have a separate function for this
			#print "one"
			result = hist(LoTitles,RefPic,histDecipher(Method))
			LoW += [result[1][1]]

		elif Method == 'hist': #if we want to use one of the histogram methods, we have a separate function for this
			#print "two"
			result1,result2,result3,result4 = hist2(LoTitles,RefPic)
			#print result1[1][1],result2[1][1],result3[1][1],result4[1][1]
			LoW += [result1[1][1],result2[1][1],result3[1][1],result4[1][1]]

		elif Method in ['mse','ssim','rms','ccv2']:
			result = matchNumNonORB(LoTitles, RefPic,LoImg, RefImg,Method)
			#print result
			LoW += [result]

		else:
			result = matchNumORB(LoTitles, RefPic,Method, LoMats, LoDists, LoKP, refKP, NumDist)
			#print result
			LoW += [result] #makes a list of winners using each method
	
	scores = Counter(LoW)
	highestScore = max(scores.values())
	winnerList = [name for name, score in scores.items() if score == highestScore]
	#print scores
	#print LoW
	#print "The image", TestPic, "is closest to:"
	#for x in winnerList:
	#	print x
	#print
	return winnerList



"""Helper methods below"""

def homography(matches,kpList1,kpList2):
	"""helper method, should return the list of homography matches given two lists of keypoints and matches between two images"""
	srcPts = np.float32([ kpList1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2) 
	dstPts = np.float32([ kpList2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC,5.0)
	#mask specifies the inlier and outlier points, while M represents the perspective transformation necessary to transform one image into the other

	matchesMask = mask.ravel().tolist() #ravel flattens the array into 1D, and then we put the mask in a list
	return len(matchesMask)

def univMode(L):
	"""helper method universalMode that takes in a list L and returns the most common element in that list"""
	data = Counter(L)
	#print data
	return data.most_common(1)[0][0]

def imGenerator(LoTitles):
	""" Helper method to take a list of image titles in and outputs a list of the ndarrays that represent the images """
	LoImg = []
	for i in LoTitles: 
		LoImg += [cv2.imread(i,0)]
	return LoImg

def orbKPGetter(LoImg,RefImg):
	"""gets the crucial ORB information for all images in a list of images, LoImg
	keeps track of the specific image for the reference image, RefImg"""
	orb = cv2.ORB()
	LoKP = [] #will hold keypoints for each image
	LoDes = [] #will hold descriptors for each image
	refKP,refDes = orb.detectAndCompute(RefImg, None)
	for Img in LoImg:
	# Computes the descriptors AND detects the keypoints all with ORB (all one function: detectAndCompute)
		#print Img
		if Img.data == RefImg.data:
			kpList = refKP
			des = refDes
		else:
			kpList, des = orb.detectAndCompute(Img, None) #these functions make lists of key points to examine
		LoKP += [kpList]
		LoDes += [des]
	return LoKP, LoDes,refKP,refDes

def orbMatchGetter(LoDes,refDes):
	"""takes in keypoints and descriptors, computes a list of matches and those matches' corresponding distances 

	BFMatcher is a brute-force descriptor"" matcher. For each descriptor in the first
	set, BFMatcher finds the closest descriptor in the second set by trying each one

	crossCheck: if crossCheck = True, then the knnMatch() (finds the k best matches for each
		descriptor from a query set) with k = 1 will return only pairs (i,j) such that for the 
	ith query descriptor, the jth descriptor in the matcher's collection is the nearest and
	vice versa"""
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	LoMats = []
	LoDists = []

	for des in LoDes:
		matches = bf.match(des, refDes) #provides a list of matches between the keypoints in the list and the reference keypoints by using the descriptors
		MatDict[Title] = matches
		LoMats += [matches]
		distances = [x.distance for x in matches] #will tell us the numeric values for the visual distances in each match
		distances.sort() #lowest, smallest distances first
		LoDists += [distances]
	#print LoMats, "space", LoDists
	return LoMats,LoDists

def histDecipher(S):
	"""helper method that takes in the name of a string S and outputs the proper tuple for use in the histogram function"""
	if S == 'histCorr':
		return ("Correlation", 0)
	elif S == 'histChi':
		return ("Chi-Squared", 1)
	elif S == 'histInter':
		return ("Intersection", 2)
	else:
		return ("Hellinger", 3)

"""Now, run any desired functions here"""

#AutoLoTitles = glob.glob('*.png')
#LoTitles = ['couch.png','couch2.png','couch3.png','crackers.png','crackers2.png','snake1.png','snake2.png','lake.png','lake2.png'] #put all images here
LoTitles2 = ['loungetest.png','lounge1.png','lounge2.png','lounge3.png','lounge4.png','lounge5.png'] #put all images here
#LoTitles3 = ['loungetest.png','lounge3.png','lounge5.png']
NumDist = 0 #put number of distances to consider here
TestPic = 'loungetest.png'
AllMethods =  ['median','mean','totaldist','halfdist','homography', 'mse','ssim','hist','rms','ccv2']
OnlyNonORB = ['mse','ssim','hist','rms']
OnlyORB = ['median','mean','totaldist','halfdist','homography']

#tester(LoTitles2,LoMeth,NumDist) #Now we compare each image in the list of images to all the others...
#singleMethTest(LoTitles,'median',NumDist) #comparing a single method...
#singlePicTest(LoTitles2,TestPic,OnlyNonORB,NumDist)#comparing a single picture...
#singlePicTest(LoTitles2,TestPic,OnlyORB,NumDist)

#print("--- %s seconds ---" % (time.time() - start_time)) #checks how much time the program runs for