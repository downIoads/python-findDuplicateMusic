import os
from multiprocessing import cpu_count
CPU_CORES = cpu_count() - 4                             # set amount of available CPU cores (was faster for me if not all cores are used)
os.environ['OPENBLAS_NUM_THREADS'] = f"{CPU_CORES}"     # fix warning that precompiled NUM_THREADS was exceeded (must happen before you import stuff like np)

import concurrent.futures
import librosa
import numpy as np
import time
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity


# getSongPathsRecursively gets any audio file of interest in any subdirectory and returns a list
def getSongPathsRecursively(directory):
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith((".wav", ".mp3", ".opus", ".flac"))]:
            files.append(os.path.join(dirpath, filename))
    return files


# splitElementsEvenlyAmongBuckets takes a list and splits the list into cpuCores many buckets as evenly as possible, returns list of lists
def splitElementsEvenlyAmongBuckets(files):    
    fileListLen = len(files)    # get total amount of files

    # evenly distribute files among CPU_CORES many buckets
    fileBuckets = []
    #       determine min amount of files per bucket
    items_per_bucket = fileListLen // CPU_CORES
    #       calculate the number of buckets that will get an extra item
    extra_buckets = fileListLen % CPU_CORES
    #       create a list to store the number of files in each bucket
    bucket_distribution = [items_per_bucket] * CPU_CORES
    #       distribute the remaining files evenly among the first few buckets
    for i in range(extra_buckets):
        bucket_distribution[i] += 1
    #       now populate fileBuckets (list of lists that contain filepaths)
    for i in bucket_distribution:
        removed_elements = files[:i]                    # take first i items from files
        fileBuckets.append(list(removed_elements))      # append them to fileBuckets as list
        del files[:i]                                   # only now actually remove these files from files
  
    return fileBuckets


# scheduleChromaAndMFCCTasks takes list of lists (files per worker thread) and assigns each sublist to a thread that calculates the chroma and mfcc of each assigned file, then returns a dictionary of filepath:(chroma, mfcc) pairs
def scheduleChromaAndMFCCTasks(fileBuckets):
    with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_CORES) as executor:
        # schedule task for each thread
        futures = [executor.submit(calculateChromasAndMFCCs, sublist) for sublist in fileBuckets] # passes sublist as argument to calculateChromas()

        # retrieve results from each task (each thread returns a dictionary)
        results = [future.result() for future in futures] # results is a list of dictionaries of filepath:(chroma, mfcc) pairs
    
        # now just unpack the list of dicts into one large dictionary
        final = {}
        for dictionary in results:
            final.update(dictionary)
        
        return final


# scheduleComparisonTasks takes a measureDict (filepath:(chroma, mfcc)) and a list of tuples (filepathA,filepathB) and returns a list of tuples that hold the paths of the estimated duplicates
def scheduleComparisonTasks(measureDict, pairBuckets):
      with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_CORES) as executor:
        # schedule task for each thread
        futures = [executor.submit(calculateSimilarity, measureDict, bucket) for bucket in pairBuckets] # passes measureDict and pairBuckets as arguments to calculateSimilarity()

        # retrieve results from each task (each thread returns a list of tuples that are the [probably] duplicate music files)
        results = [future.result() for future in futures]
      
        # unpack list of lists of tuples into one big list that holds all duplicate tuples
        final = [item for sublist in results for item in sublist] # item stands for tuple, tuple is a protected name in python
        
        return final


# calculateChromasAndMFCCs takes a list of filepaths and returns a dictionary of filepath:(chroma,mfcc) pairs
def calculateChromasAndMFCCs(myFilelist):
    threadDict = dict()
    for curPath in myFilelist:
        y, sr = librosa.load(curPath)
        # calculate chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # calculate mfcc (Mel-frequency cepstral coefficients) and the mean over time
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        meanMFCC = np.mean(mfcc, axis=1)
        # add data to dict
        threadDict[curPath] = (chroma, meanMFCC)

    return threadDict
    

# calculateSimilarity takes a dict [filepath:(chroma, mfcc) pairs] and a list of tuples [filepath:OtherFilepath] and checks whether any filepair tuple stands for duplicate audio files, returns a list of tuples that probably are duplicates
def calculateSimilarity(measureDict, tupleList):
    # chromas have been precalculated, so just [4head] loop over pairBuckets and get each chroma from measureDict and then do cosine similarity calculation
    possibleDuplicates = []
    for t in tupleList: # t = (a,b) where a and b are filepaths
        # get chromas
        aChroma = measureDict[t[0]][0]
        bChroma = measureDict[t[1]][0]

        # get mfcc
        aMFCC = measureDict[t[0]][1]
        bMFCC = measureDict[t[1]][1]

        # calculate mfcc similarity
        mfccScore = cosine_similarity([aMFCC], [bMFCC])[0][0]

        # calculate chroma similarity
        similarity = cosine_similarity(aChroma.T, bChroma.T)
        chromaScore = np.max(similarity)
        if chromaScore > 0.99996: # finetuned parameters, very important
            if chromaScore > 0.99999 or mfccScore > 0.96:
        # chroma score: should always be lower than 0.999999, good strict score is around 0.999962 (can be lower if you consider other scores like mfcc  too)
                
        # mfcc score: should be larger than 0.965
                possibleDuplicates.append((t[0], t[1], chromaScore, mfccScore)) # add chroma similiarity score as third value in tuple, and mfcc similarity as fourth value

    return possibleDuplicates


def main():
    start_time = time.time()                        # measure execution time (can be quite slow)
    rootdir = '.\\'                                 # root directory that will be recursively searched for music files in any subfolders 

    # get filelist
    fileList = getSongPathsRecursively(rootdir)
    fileListLength = len(fileList)                  # remember its length now as it gets nuked later
    
    # calculate chromas and mfccs in parallel
    fileBuckets = splitElementsEvenlyAmongBuckets(fileList)
    measureDict = scheduleChromaAndMFCCTasks(fileBuckets)   # returns a dicts that holds all filepath:(chroma, mfcc) pairs
    if fileListLength != len(measureDict):
        print("Sth went wrong. Aborting.")
        return
    
    # get file tuple list (pairwise song comparison, tuple (a,b) and (b,a) considered identical)
    keys = list(measureDict.keys())
    uniqueKeyPairsList = list(combinations(keys, 2)) # list of tuples (a,b) where a and b are filepaths
    
    # find duplicate songs
    pairBuckets = splitElementsEvenlyAmongBuckets(uniqueKeyPairsList)
    duplicateList = scheduleComparisonTasks(measureDict, pairBuckets) # returns list of tuples

    # print results
    if len(duplicateList) == 0:
        print("No duplicates found!\nExecution time:", time.time() - start_time)
        return
    for dup in duplicateList:
        print(f'Suspected duplicate files (chroma score {dup[2]}, mfcc score {dup[3]}):\n\t{dup[0]}\n\t{dup[1]}\n')
    print(f"\n----\nTotal amount of suspected duplicates: {len(duplicateList)}\nExecution time: {time.time() - start_time} seconds")


if __name__ == '__main__':      # important to not mess with multiple threads
    main()

# Runtime:  - 92 sec for 27 songs (386 MB in total) on i9 12900k, so very slow.. finds all duplicate pairs and one false positive pair
#           -  7 sec for 10 songs (200 MB in total)

# Notes: By "duplicate" songs I mean this tool is able to find different audio quality files (mp3 vs flac) that might have
#        different lengths (1:30 min TV version and 4:45 full version) etc. So it's pretty impressive, but sadly very slow due to n*(n-1)/2 #        unique pairs that need to compared.