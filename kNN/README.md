# Use kNN ML to identify handwriting digits

kNN (*k-Nearest Neighbors*) classification algorithm is probably the simplest ML algorithm yet effective to start your journey to \ML. I am going to develop a kNN algorithm to identify handwriting digits in text format. An example handwriting digits which is formatted as text.

![](https://i.imgur.com/vn1ts9n.jpg)

## k-Nearest Neighbors

k-Nearest Neighbors can be classified as the supervised learning algorithm. To make it work, we need to have samples dataset or training dataset. Each sample in training dataset is a feature vector (特征值).

Take our handwriting training dataset as an example, each digit has been prepared a variety of samples. For example, the digit 0, we got 188 samples with each sample to represent a variation of the handwriting of 0. Each digit sample is saved as a text file.

We also need to label or class each sample. In our case, the labels are digits from 0 to 9 and assign to each digit sample in our handwriting training set.

When we're given a new digit sample text file, we ask our kNN algorithm to identify the digit in it and label it as a digit in class 0 to 9.

The idea of k-NN is to take the new sample and then convert it to a feature vector. In our case, the digit is a 32x32 image formatted as 0,1 text file. To convert it as a feature vector, we will load image file into a vector like [0, [1,0,…0,1]], the length of the inner vector is 1024 (32x32).

We then measure the distance of this new sample vector to every sample vector in the training set. It really is computing intensive so k-NN isn't quite efficient for large training set use case.

The measurement basically is an extension of Pythagorean theorem （勾股定律). As I believe knowing how kNN measurement work in mathematical isn't really matters for understanding kNN, I will assume we all know how kMM measurement works. We then take the most similar, in the sense of distance, pieces of sample data (the nearest neighbors) and look at their labels (0 to 9). We look at the top k most similar pieces of sample data from our known dataset; this is where the k comes from. (k is an integer and it's usually less than 20.) Lastly, we take a majority vote from the k most similar pieces of data, and the majority is the new class in 0 to 9 we assign to the new data we were asked to classify.

## Prepare: converting images into test vectors


The images are stored in two directories in the [github](https://github.com/jianhuabi/ml-demo/tree/master/kNN) source code. The **training-Digits** directory contains about 2,000 examples similar to those in the demo picture. There are roughly 200 samples for each digit. The **testDigits** directory contains about 900 examples. We’ll use the *trainingDigits* directory to train our **kNN-classifier** and *testDigits* to test it. There’ll be no overlap between the two groups. Feel free to take a look at the files in those folders.

We’ll take the 32x32 matrix that is each binary image and make it a 1x1024 vector. After we do this, we can apply it to our kNN classifier.

The following code is a small function called **img2vector**, which converts the image to a vector. The function creates a 1x1024 *NumPy* array, then opens the given file, loops over the first 32 lines in the file, and stores the integer value of the first 32 characters on each line in the NumPy array. This array is finally returned.

```python=
def img2vector(filename):
    returnVect = zeros((1,1024)) //Init a empty vector, [0, [0,0...,0]]
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
```

Try out the *img2vector* code with the following commands in the **Jupyter Notebook**, and compare the results to a file opened with a text editor: 

![](https://i.imgur.com/Jk6fnRc.png)

##  k-Nearest Neighbors algorithm 
Pseudocode for this function would look like this:

```
For every vector/sample in our training dataset:

calculate the distance between inX (new sample vec) and the current vector/sample in training dataset

sort the distances in increasing order

take k items with lowest distances to inX

find the majority class among these items

return the majority class as our prediction for the class of inX
```

The Python code for the **classify0() function** is in the following listing.

```python=
def classify0(inX, dataSet, labels, k):
    # dataSet is [m x 1] matrix, dataSetSize is the samples count in training set. 
    dataSetSize = dataSet.shape[0]
    # next four line calculate distance of new sample to all samples in training set.
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    # sort distance
    sortedDistIndicies = distances.argsort()     
    classCount={}   
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]
```

* **inX** : new sample or new digit we want to recognize.
* **dataSet**: training dataset
* **labels**: class of training dataset, in 0 ~ 9
* **k**: number of nearest neighbors, 3 for example

## Test: kNN on handwritten digits 
The function shown below, *handwritingClassTest()*, is a self-contained function that tests out our classifier. 

```python=
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, \
        hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: \
        %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

```

- In this function, you get the contents from the **trainingDigits directory** as a list. 
- Then you see how many files are in that directory and call this **m**. 
- Next, you create a training matrix with **m** rows and **1024 columns** to hold each image as a single row. 
- You parse out the **class number from the filename**. The filename is something like 9_45.txt, where 9 is the class number and it is the 45th instance of the digit 9. 
- You then put this class number in the *hwLabels* vector and load the image with the function **img2vector** discussed previously. 
- Next, you do something similar for all the files in the **testDigits directory**, but instead of loading them into a big matrix, you test each vector individually with our **classify0 function**. 

I am running this function via *Jupyter Notebook* and results as something in below.

![](https://i.imgur.com/KRhbbUN.png)

This classifier shows 1.2% error rate which is acceptable. 

You can vary **k** to see how this changes. You can also modify the **handwritingClassTest** function to randomly select training examples. That way, you can vary the number of training examples and see how that impacts the error rate. 

## Summary 
The k-Nearest Neighbors algorithm is a simple and effective way to classify data. But the algorithm has to carry around the full dataset; for large datasets, this implies a large amount of storage. In addition, you need to calculate the distance measurement for every piece of data in the database, and this can be cumbersome. 

If you want to execute this program, you can find sample datasets and source code in below github.

https://github.com/jianhuabi/ml-demo/tree/master/kNN

