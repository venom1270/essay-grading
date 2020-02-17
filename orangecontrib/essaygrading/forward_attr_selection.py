from orangecontrib.essaygrading.widgets.OWScore import quadratic_weighted_kappa
from orangecontrib.essaygrading.widgets.OWAttributeSelection import calculateAttributes
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

from Orange.data import Table, Domain, ContinuousVariable
from Orange.regression import RandomForestRegressionLearner
from Orange.regression import LinearRegressionLearner

# table = Table.from_file("datasets/FAS/set1_train_2.tab")
#table = Table.from_file("datasets/FAS/set1_train_2.tab")
table = Table.from_file("datasets/FAS/set8_flair.tab")

scores = np.array(table.Y)
ALL_ATTRIBUTES = np.array([x for x in table.X])
ALL_ATTRIBUTES = np.nan_to_num(ALL_ATTRIBUTES)
print(table)

print(ALL_ATTRIBUTES.shape)

scaler = preprocessing.MinMaxScaler()
#ALL_ATTRIBUTES = scaler.fit_transform(ALL_ATTRIBUTES)

# ALL_ATTRIBUTES = preprocessing.normalize(ALL_ATTRIBUTES.transpose()).transpose()

print(ALL_ATTRIBUTES)
print(max(ALL_ATTRIBUTES[:, 5]))
print(min(ALL_ATTRIBUTES[:, 5]))

k = 10
chunk = int(len(ALL_ATTRIBUTES) / k)

USE_SKLEARN_RFE = False

if USE_SKLEARN_RFE:
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(ALL_ATTRIBUTES, scores)
    selector = RFE(rf, 50, step=1)
    selector.fit(ALL_ATTRIBUTES, scores)
    print(selector.support_)
    print(selector.ranking_)

    for i in range(len(selector.support_)):
        if selector.support_[i]:
            print(table.domain.attributes[i])

    exit()

attributes = np.array([ALL_ATTRIBUTES[:, 0]]).transpose()
kept_attributes = []
avg_kappa = 0

train = []
train_scores = []
test = []
test_scores = []

for j in range(10):
    start = j * chunk
    end = min((j + 1) * chunk, len(ALL_ATTRIBUTES))
    train.append(np.concatenate((ALL_ATTRIBUTES[:start], ALL_ATTRIBUTES[end:])))
    train_scores.append(np.concatenate((scores[:start], scores[end:])))
    test.append(ALL_ATTRIBUTES[start:end])
    test_scores.append(scores[start:end])



for iter in range(len(ALL_ATTRIBUTES[0])):

    print("STARTING NEW ITERATION: " + str(iter) + "/" + str(len(ALL_ATTRIBUTES[0])) + " (max)")

    added_attribute = None
    best_score = -1
    best_i = -1
    curr_kappa = 0

    for i in range(len(ALL_ATTRIBUTES[0])):

        if i in kept_attributes:
            continue

        print("ATTRIBUTE " + str(i) + " / " + str(len(ALL_ATTRIBUTES[0])))
        # print(ALL_ATTRIBUTES[:, i])

        kappas = []
        #attributes = np.concatenate((attributes, np.array([ALL_ATTRIBUTES[:, i]]).transpose()), axis=1)
        taken_attributes = kept_attributes + [i]

        #attributes = np.concatenate((attributes, np.array([ALL_ATTRIBUTES[:, i]]).transpose()), axis=1)
        #attributes = np.take(ALL_ATTRIBUTES, taken_attributes, axis=1)


        for j in range(k):
            # print("J = " + str(j))
            #start = j * chunk
            #end = min((j + 1) * chunk, len(attributes))
            #train = np.concatenate((attributes[:start], attributes[end:]))
            #train_scores = np.concatenate((scores[:start], scores[end:]))
            #test = attributes[start:end]
            #test_scores = scores[start:end]
            #To zdaj naredim samo enkrat zgoraj


            # print(train.shape)
            # print(test.shape)

            train_attributes = np.take(train[j], taken_attributes, axis=1)
            test_attributes = np.take(test[j], taken_attributes, axis=1)

            #rf = LinearRegressionLearner()  # (n_estimators=100, )#, min_samples_split=max(int(len(attributes[0])/3), 2))
            rf = RandomForestRegressionLearner(n_estimators=100, bootstrap=False, )#, min_samples_split=max(int(len(attributes[0])/3), 2)))

            # print("Training...")
            rf = rf.fit(train_attributes, train_scores[j])

            # print("Predicting...")
            predictions = rf.predict(test_attributes)

            # print("Results...")
            kappas.append(quadratic_weighted_kappa(np.round(predictions), test_scores[j]))

        print("FINISHED")
        print(kappas)
        curr_kappa = sum(kappas) / k
        print(str(curr_kappa) + " >= " + str(avg_kappa))

        if curr_kappa >= best_score:
            best_score = curr_kappa
            best_i = i

        attributes = attributes[:, :-1]

        '''if curr_kappa >= avg_kappa:



            avg_kappa = curr_kappa
            kept_attributes.append(i)
            print("KEEPING ATTRIBUTE " + str(i) + "   ### " + str(avg_kappa))
            added_attribute = i
            break
        else:
            # Remove attribute
            attributes = attributes[:, :-1]'''

    '''if added_attribute is None:
        print("Added attribute is None; BREAK")
        break'''

    if best_score >= avg_kappa:
        print("Attribute " + str(best_i) + " had highest score: " + str(best_score) + " | avg: " + str(avg_kappa))
        avg_kappa = best_score
        attributes = np.concatenate((attributes, np.array([ALL_ATTRIBUTES[:, best_i]]).transpose()), axis=1)
        kept_attributes.append(best_i)

        print(kept_attributes)

    else:
        print("No attribute increased score; BREAK;")
        break

print("KEPT ATTRIBUTES: ")
print(kept_attributes)
print(sorted(kept_attributes))
print(len(kept_attributes))
for i in kept_attributes:
    print(table.domain.attributes[i])

'''

# rf = RandomForestRegressionLearner(n_estimators=100)
KEPT ATTRIBUTES: 
[1, 2, 3, 4, 10, 11, 13, 20, 59, 79, 83]
kappa = 0.78neki



# rf = RandomForestRegressionLearner(n_estimators=100, min_samples_split=max(int(len(attributes[0])/3), 2))
[1, 2, 3, 4, 11, 12, 13, 30, 74]
kappa= 0.785676286129781





RFE:
[ True  True False  True False  True  True False False  True  True  True
  True False  True False  True False  True False False  True False  True
  True False  True  True False  True  True False False  True False False
  True  True False False False False  True False False False False False
 False False  True False False False False  True False  True False  True
 False  True False False False False False False False False  True  True
  True False  True  True False  True False  True  True False False  True
  True  True  True  True False  True  True False False  True  True  True
 False  True False  True  True  True]
[ 1  1 17  1 44  1  1 28 34  1  1  1  1  8  1 21  1 46  1 13 33  1 16  1
  1 27  1  1  9  1  1  2 36  1 37 47  1  1 38 43 52  7  1 15 23 51 39 45
  5  6  1 41 48 22 49  1 50  1 10  1 12  1 14 32 53 25  3 40 30 18  1  1
  1 19  1  1 20  1 24  1  1 11 26  1  1  1  1  1 35  1  1  4 29  1  1  1
 42  1 31  1  1  1]



 --


 [ True  True  True  True False  True  True False False  True  True  True
 False False  True False  True False  True False False  True False  True
  True False  True  True False  True  True False False  True False False
  True  True False False False False  True False False False False False
 False  True  True False False False False  True False  True False False
 False  True False False False False False False False False  True  True
  True False False  True  True  True  True  True False False  True False
 False  True  True  True  True  True  True False False  True  True  True
 False  True False  True  True  True]
[ 1  1  1  1 43  1  1 34 29  1  1  1  2  8  1 18  1 47  1  9 35  1 27  1
  1 10  1  1 15  1  1 13 37  1 41 45  1  1 38 44 52 14  1  7 19 53 36 46
 11  1  1 42 48 20 49  1 50  1 17  4 16  1  3 31 51 25 12 40 33 23  1  1
  1 24 32  1  1  1  1  1 22  6  1 21 26  1  1  1  1  1  1  5 39  1  1  1
 30  1 28  1  1  1]
numberOfCharacters
numberOfWords
numberOfShortWords
numberOfLongWords
averageWordLength
numberOfSentences
mostFrequentSentenceLength
averageSentenceLength
numberOfDifferentWords
fleschReadingEase
daleChallReadabilityFormula
simpleMeasureOfGobbledygook
nominalRatio
guiraudsIndex
yulesK
hapaxLegomena
advancedGuiraudIndex
sentenceStructureTreeHeight
correctVerbForm
pos_DT
pos_IN
pos_JJ
pos_NN
pos_PRP$
pos_RB
pos_TO
pos_VB
pos_VBP
cosineTopEssaySimilarityAverage
cosinePattern
cosineSumOfCorrelationValues
maxDistanceNeighbouringPointsEuc
indexDistanceNeighbouringPointsEuc
avgDistanceNeighbouringPointsCos
minDistanceNeighbouringPointsCos
maxDistanceNeighbouringPointsCos
maxDistanceAnyPointsEuc
clarkEvansNearestNeighbour
avgDistanceNearestNeighbour
cumulativeFrequencyDistribution
avgDistanceCentroidEuc
minDistanceCentroidEuc
maxDistanceCentroidEuc
minDistanceCentroidCos
maxDistanceCentroidCos
indexDistanceCentroidCos
relativeDistance
moransI
gearysC
getissG





'''

'''
****** 
DATASET 1_2.tab
******


*** RFE ***
[ True  True  True  True  True False  True False False False False  True
  True  True False False  True  True False  True False False  True False
 False  True  True False False  True False  True  True  True False  True
 False False  True  True False False False False  True False False False
 False False False  True  True False False False False False False  True
 False  True False  True False False False False  True False False False
  True  True  True  True False  True False False  True  True  True False
  True  True False  True  True  True False  True  True False False  True
 False  True False  True False  True  True  True]
[ 1  1  1  1  1 43  1 17 22 12  4  1  1  1 11  2  1  1 50  1 26 30  1 41
 33  1  1  3  7  1 20  1  1  1 36  1 40 51  1  1 38 44 55  8  1 14 25 54
 39 47  5  1  1 42 49 24 48  6 52  1 21  1 19  1 10 34 53 29  1 45 28 16
  1  1  1  1 31  1 23 13  1  1  1 18  1  1 32  1  1  1 35  1  1  9 37  1
 15  1 46  1 27  1  1  1]
numberOfCharacters
numberOfCharactersNoSpaces
numberOfWords
numberOfShortWords
numberOfLongWords
averageWordLength
averageSentenceLength
numberOfDifferentWords
numberOfStopwords
fleschKincaidGradeLevel
daleChallReadabilityFormula
simpleMeasureOfGobbledygook
nominalRatio
guiraudsIndex
yulesK
advancedGuiraudIndex
sentenceStructureTreeHeight
correctVerbForm
pos_CC
pos_DT
pos_IN
pos_JJ
pos_NN
pos_PRP$
pos_RB
pos_VB
pos_VBG
pos_VBP
numberOfSpellcheckingErrors
cosineTopEssaySimilarityAverage
cosinePattern
cosineSumOfCorrelationValues
avgDistanceNeighbouringPointsEuc
maxDistanceNeighbouringPointsEuc
minDistanceNeighbouringPointsCos
maxDistanceNeighbouringPointsCos
indexDistanceNeighbouringPointsCos
maxDistanceAnyPointsEuc
avgDistanceAnyPointsCos
clarkEvansNearestNeighbour
avgDistanceNearestNeighbour
cumulativeFrequencyDistribution
minDistanceCentroidEuc
maxDistanceCentroidEuc
minDistanceCentroidCos
indexDistanceCentroidCos
relativeDistance
moransI
gearysC
getissG


*** MANUAL ***

FINISHED
[0.8314259307858195, 0.7679269882659714, 0.8287030277287389, 0.7897044181242663, 0.7951114750552355, 0.7841583300853145, 0.8040578458881934, 0.6972020514903772, 0.8139388166602755, 0.7522616562282534]
0.7864490540312445 >= 0.7876473428529129
KEPT ATTRIBUTES: 
[1, 2, 3, 4, 7, 9, 11, 12, 13, 15, 23, 40, 76]
numberOfCharactersNoSpaces
numberOfWords
numberOfShortWords
numberOfLongWords
numberOfSentences
numberOfLongSentences
averageSentenceLength
numberOfDifferentWords
numberOfStopwords
fleschReadingEase
lexicalDiversity
pos_JJR
minDistanceNeighbouringPointsEuc


'''

'''
set1_train2.tab
KEPT ATTRIBUTES: 
[0, 1, 2, 3, 4, 5, 7, 12, 6, 11, 13, 8, 32, 49, 14, 25, 74]
numberOfCharacters
numberOfCharactersNoSpaces
numberOfWords
numberOfShortWords
numberOfLongWords
mostFrequentWordLength
numberOfSentences
numberOfDifferentWords
averageWordLength
averageSentenceLength
numberOfStopwords
numberOfShortSentences
correctVerbForm
pos_POS
gunningFogIndex
guiraudsIndex
cosineSumOfCorrelationValues


 0.79eki se mi zdi




 0.7843160014472802 >= 0.7908258418926252
Added attribute is None; BREAK
KEPT ATTRIBUTES: 
[0, 1, 2, 3, 4, 5, 6, 8, 12, 9, 7, 13, 10, 20, 69, 62]
numberOfCharacters
numberOfCharactersNoSpaces
numberOfWords
numberOfShortWords
numberOfLongWords
mostFrequentWordLength
averageWordLength
numberOfShortSentences
numberOfDifferentWords
numberOfLongSentences
numberOfSentences
numberOfStopwords
mostFrequentSentenceLength
lix
numberOfCapitalizationErrors
pos_VBN
'''

'''
0.549...
set2A_train_tfidf.set
numberOfCharacters
numberOfCharactersNoSpaces
numberOfWords
numberOfDifferentWords
numberOfShortWords
numberOfSentences
numberOfLongWords
mostFrequentSentenceLength
averageWordLength
numberOfStopwords
pos_CD
pos_WRB
simpleMeasureOfGobbledygook

Orange WK: 0.663874


0.5486850959052247 >= 0.5587769421326065
NORMALIZED
numberOfCharacters
numberOfCharactersNoSpaces
numberOfWords
mostFrequentWordLength
numberOfDifferentWords
numberOfLongWords
numberOfSentences
averageWordLength
numberOfShortWords
numberOfCapitalizationErrors
gunningFogIndex
pos_VBP
numberOfShortSentences
automatedReadabilityIndex
sentenceStructureTreeHeight
pos_IN
simpleMeasureOfGobbledygook
mostFrequentSentenceLength
numberOfLongSentences
pos_JJR


Orange WK: 0.689733



-- NORMALIZED, nbrez omejitev (razen n=100)
0.5275334520898579 >= 0.54406018591143
Added attribute is None; BREAK
KEPT ATTRIBUTES: 
[0, 1, 2, 12, 4, 3, 5, 7, 6, 11]
numberOfCharacters
numberOfCharactersNoSpaces
numberOfWords
numberOfDifferentWords
numberOfLongWords
numberOfShortWords
mostFrequentWordLength
numberOfSentences
averageWordLength
averageSentenceLength



--- ne nromalized; brez omejitev
0.5454701992678507 >= 0.5658352655291223
Added attribute is None; BREAK
KEPT ATTRIBUTES: 
[0, 1, 2, 9, 12, 7, 4, 6, 3, 8, 10, 16, 18, 69, 31, 48, 19]
numberOfCharacters
numberOfCharactersNoSpaces
numberOfWords
numberOfLongSentences
numberOfDifferentWords
numberOfSentences
numberOfLongWords
averageWordLength
numberOfShortWords
numberOfShortSentences
mostFrequentSentenceLength
fleschKincaidGradeLevel
automatedReadabilityIndex
numberOfCapitalizationErrors
sentenceStructureTreeHeight
pos_PDT
simpleMeasureOfGobbledygook
'''

'''

RANDOM FOREST (100) s kept attributes, brez normalizacije

[0.6928879310344827, 0.6667771883289125, 0.6530612244897958, 0.6475346687211093, 0.6814159292035399, 0.7006919487206995, 0.703770197486535, 0.7708757637474541, 0.7067594433399602, 0.7443762781186094]
0.6968150573191098 >= 0.7120839985753566
No attribute increased score; BREAK;
KEPT ATTRIBUTES: 
[12, 74, 75, 70, 72, 7, 71, 73]
[7, 12, 70, 71, 72, 73, 74, 75]
8
numberOfDifferentWords
cosineSumOfCorrelationValues
avgDistanceNeighbouringPointsEuc
numberOfPunctuationErrors
cosineTopEssaySimilarityAverage
numberOfSentences
scoreCosineSimilarityMax
cosinePattern



---- Z NORMALIZACIJO
0.6917065705633654 >= 0.7125073991394089
No attribute increased score; BREAK;
KEPT ATTRIBUTES: 
[12, 74, 7, 25, 88, 28, 75, 26]
[7, 12, 25, 26, 28, 74, 75, 88]
8
numberOfDifferentWords
cosineSumOfCorrelationValues
numberOfSentences
guiraudsIndex
avgDistanceNearestNeighbour
hapaxLegomena
avgDistanceNeighbouringPointsEuc
yulesK


'''


'''
SET1 brez normalizacije, RF 100
0.8367150627693467 >= 0.8430332068200113
No attribute increased score; BREAK;
KEPT ATTRIBUTES: 
[12, 0, 13, 103, 102, 63, 38]
[0, 12, 13, 38, 63, 102, 103]
7
numberOfDifferentWords
numberOfCharacters
numberOfStopwords
getissG
gearysC
pos_VBP
pos_IN

'''

'''
SET 1 brez nromalizacije:
0.8380983087699387 >= 0.8392046167515128
KEPT ATTRIBUTES: 
[12, 0, 13, 63, 102, 74, 19]
[0, 12, 13, 19, 63, 74, 102]
7
numberOfDifferentWords
numberOfCharacters
numberOfStopwords
pos_VBP
gearysC
cosineSumOfCorrelationValues
simpleMeasureOfGobbledygook


----------------------------------
ISTO ŠE ENKRAT... preveč random je to
0.8360311207116018 >= 0.8415334874271302
No attribute increased score; BREAK;
KEPT ATTRIBUTES: 
[12, 1, 95, 13, 72, 37]
[1, 12, 13, 37, 72, 95]
6
numberOfDifferentWords
numberOfCharactersNoSpaces
minDistanceCentroidCos
numberOfStopwords
cosineTopEssaySimilarityAverage
pos_FW



TO JE ZDEJ Z GOZODVI n_estimators=1000
0.838912772826405 >= 0.8426979174044854
No attribute increased score; BREAK;
KEPT ATTRIBUTES: 
[12, 1, 95, 5, 13, 72]
[1, 5, 12, 13, 72, 95]
6
numberOfDifferentWords
numberOfCharactersNoSpaces
minDistanceCentroidCos
mostFrequentWordLength
numberOfStopwords
cosineTopEssaySimilarityAverage



'''


''''
SET 8 FLAIR n_estimators=1000
0.7253585068963597 >= 0.7299612298464113
No attribute increased score; BREAK;
KEPT ATTRIBUTES: 
[12, 21, 69, 7, 72, 68, 25, 4, 10]
[4, 7, 10, 12, 21, 25, 68, 69, 72]
9
numberOfDifferentWords
wordVariationIndex
numberOfCapitalizationErrors
numberOfSentences
cosineTopEssaySimilarityAverage
numberOfSpellcheckingErrors
guiraudsIndex
numberOfLongWords
mostFrequentSentenceLength


'''