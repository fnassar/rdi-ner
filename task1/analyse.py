import io
import re


def non_arabic_ratio(line):
    arabic_chars = re.findall('[\u0600-\u06FF]+', line)
    arabic_len = sum(len(word) for word in arabic_chars)
    return (len(line) - arabic_len) / len(line)


def main():

    # add num eng letters and num arabic letters
    wordList = [[0 for x in range(4)] for y in range(1)]
    wEngNum = 0
    wArabNum = 0
    # has sentence, number of words in sentence, eng char in sentence, arabic char in sentence, number of entities in sentence,
    sentenceList = [[0 for x in range(2)] for y in range(1)]
    wEngNum = 0
    wArabNum = 0
    minWnum = 0
    maxWnum = 99

    sentence = ""
    collectedSentence = ""
    types = []
    i = 0

    # has word, type, length, number of entities
    numentities = 0
    numWords = 0
    numChars = 0
    numSentences = 0
    minLen = 0
    maxLen = 0
    n = 0

    # Open file
    with io.open("ANERCorp_Benajiba.txt", 'r', encoding='utf-8') as file1:
        # reading each line
        for line in file1:
            # remove \n
            line2 = line.strip('\n')
            line2 = line2.split(' ')

            # analyze line
            if ((len(line2[0]) == 1)):
                if (((ord(line2[0]) < 0x064A and ord(line2[0]) > 0x0620) or (ord(line2[0]) > 0x30 and ord(line2[0]) < 0x39))):
                    # count word details
                    wordList.append(
                        [line2[0], line2[1], len(line)-2, len(line2[0])])
                    wArabNum += len(line2[0])
                    wEngNum += len(line2[1])
                    if (maxWnum < len(line2[0])):
                        maxWnum = len(line)-2
                    if (minWnum > len(line2[0])):
                        minWnum = len(line)-2
                    i += 1

            # analyze sentence
            if line2[0] == '.':
                # add var for these TODO: add to sentenceList and fix variables
                sentenceList.append(
                    [sentence, numentities, numWords, numChars, types])
                # CALC RATIO
                if(len(collectedSentence) > 0):
                    ratio = non_arabic_ratio(collectedSentence)
                    print("Sentence: ", collectedSentence, "Ratio: ", ratio)
                collectedSentence = ""
                sentence = ""
                numentities = 0
                numChars = 0
                numWords = 0
                types = []
                numSentences += 1
                if (maxLen < numWords):
                    maxLen = numWords
                if (minLen > numWords):
                    minLen = numWords

            # collected sentence data

            else:
                if line2[1] != 'O':
                    numentities += 1
                numWords += 1
                numChars += len(line2[0])
                types.append(line2[1])
                sentence += line2[0]
                collectedSentence += line.strip('\n')
                n += 1
        file1.close()
    # word
    print("Number of words: ", len(wordList))
    print("Number of words, ar: ", wArabNum, "Number of words, en: ", wEngNum)
    print("Average word length: ", (wArabNum+wEngNum)/len(wordList))
    print("Max word length: ", maxWnum, "Min word length: ", minWnum)
    # calc std

    # sentence


# calc

main()
