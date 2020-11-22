import os
import sys
import timeit
import re
import string
import nltk
import xml.sax
from collections import *
import heapq
from tqdm import tqdm
import threading

# All the global variables used in the code
dictionary={}
pages=0
files=0
inverted_index=defaultdict(list)
offset=0

# All the stopwords are listed below and are taken from list of stopwords in nltk
stopwords=set(["a", "about", "above", "above", "across", "after", "afterwards", "again", "against",
 "all", "almost", "alone", "along", "already", "also","although","always","am","among",
  "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything",
  "anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", 
  "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond",
   "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de",
    "describe", "detail", "do", "done", "down","due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", 
    "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", 
    "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", 
    "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", 
    "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", 
    "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", 
    "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither",
     "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often",
    "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", 
    "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show",
    "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", 
    "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", 
    "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", 
    "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we",
     "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", 
     "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without",
      "would", "yet", "you", "your", "yours", "yourself", "yourselves"])



# Text Preprocessing Steps
# ********************************************************************************************#


# I am not using Stemmer for time benefits as using Stemmer was consuming a lot of time.
#  If you want to use kindly uncomment the below lines

#from nltk.stem.porter import *   
#ps=PorterStemmer() 
# Function module to do stemming
# def stemming(content):
#     content=[ps.stem(word) for word in content]
#     return content



# Function Module to do Tokenization
# 1) Removing special characters from text
# 2) Removing URLs from text
# 3) Removing Html elements from text
def tokenize(text):
    text=text.encode("ascii", errors="ignore").decode()
    # removing special characters
    text = re.sub(r'[^A-Za-z0-9]+', r' ', text)
    # removing urls
    text=re.sub(r'http[^\ ]*\ ', r' ', text) 
    # removing html entities
    text = re.sub(r'&nbsp;|&lt;|&gt;|&amp;|&quot;|&apos;', r' ', text) 
    text=text.split() # Dividing into words
    return text

# Function module to remove stop-words
def StopWords_removal(content):
    content_modified=[]
    for word in content:
        if word not in stopwords:
            content_modified.append(word)
    return content_modified

#**********************************************************************************************#

# Function Modules to extract Titles, references,categories,Body,InfoBox and Links

  #  EXTRACTING THE TITLE
def  extractTitle(title):
  
    title=title.lower() #Case Folding for the title
    title=tokenize(title)
    title=StopWords_removal(title)
    # title=stemming(title) # Uncomment it if you want to include stemming in your preprocessing steps
    return title

# Extracting Body
def extractBody(text):
    text=text.lower()
    temp = re.sub(r'\{\{.*\}\}', r' ', text)
    body=tokenize(temp)
    body=StopWords_removal(body)
    # body=stemming(body)
    return body

 # EXTRACTING INFO
def extractInfo(text):
    text=text.lower()
    content_splitted=text.split('\n')
    flag=False
    info=[]
   
    for word in content_splitted:
        if re.match(r'\{\{infobox', word):
            temp=re.sub(r'\{\{infobox(.*)', r'\1',word)
            info.append(temp)
            flag=True
        elif flag== True:
            if word == "}}":
                flag= False
                continue
            info.append(word)
    info = tokenize(' '.join(info))
    info = StopWords_removal(info)
    # info=stemming(info)
    return info


# EXTRACTING REFERENCES
def extractReferences(text):
    content_splitted = text.split('\n')
    references= []
    for word in content_splitted:
            if re.search(r'<ref', word):
                references.append(re.sub(r'.*title[\ ]*=[\ ]*([^\|]*).*', r'\1', word))

    references=tokenize(' '.join(references))
    references=StopWords_removal(references)
    # references=stemming(references)
    return references


#EXTRACTING LINKS
def extractLinks(text):
    content_splitted = text.split('\n')
    links = []
    for word in content_splitted:
            if re.match(r'\*[\ ]*\[', word):
                links.append(word)
        
    links = tokenize(' '.join(links))
    links = StopWords_removal(links)
    # links=stemming(links)
    return links

 #EXTRACTING CATEGORIES
def extractCategories(text):
    content_splitted=text.split('\n')
    categories = []
    for word in content_splitted:
        if re.match(r'\[\[category', word):
            temp=re.sub(r'\[\[category:(.*)\]\]', r'\1',word)
            categories.append(temp)
        
    categories=tokenize(' '.join(categories))
    categories=StopWords_removal(categories)
    # categories=stemming(categories)
    return categories

#*************************************************************************************#







class writeThread(threading.Thread):
    def __init__(self, field, data, offset, count):
        threading.Thread.__init__(self)
        self.field = field
        self.offset = offset
        self.data = data
        self.count = count
    def run(self):
        
        f_name =  './files/' + self.field + str(self.count) + '.txt'
        with open(f_name, 'w') as f:
            f.write('\n'.join(self.data))
        f_name = './files/supu' +self.field + str(self.count)+ '.txt'
        with open(f_name, 'w') as f:
            f.write('\n'.join(self.offset))









# Writing into file
def writeIntoFile(inverted_index, files,dictionary,offset):
    data_offset = []    
    data = []
    previous_offset = offset
    for key in sorted(dictionary):
        temp = str(key) + ' ' + dictionary[key].strip()
        size_of_temp=len(temp)
        if(size_of_temp):
            previous_offset = 1 + previous_offset + size_of_temp
        else:
            previous_offset = 1 + previous_offset
        data.append(temp)
        data_offset.append(str(previous_offset))
    f_name = './files/titleOffset.txt'
    try:
        with open(f_name, 'a') as f:
            f.write('\n'.join(data_offset))
            f.write('\n')
    except:
        os.mkdir('files')
        with open(f_name, 'a') as f:
            f.write('\n'.join(data_offset))
            f.write('\n')
    f_name = './files/title.txt'
    with open(f_name, 'a') as f:
        f.write('\n'.join(data))
        f.write('\n')

    data = []
    for key in sorted(inverted_index.keys()):
        postings = inverted_index[key]
        string = key + ' '
        string = string + ' '.join(postings)
        data.append(string)
    file_name = './files/inverted_index' 
    f_name = file_name + str(files) + '.txt'
    with open(f_name, 'w') as f:
        f.write('\n'.join(data))
    return previous_offset









def finalWrite(data,finalCount,offsetSize):
    offset=[]
    title=defaultdict(dict)
    body=defaultdict(dict)
    info=defaultdict(dict)
    category=defaultdict(dict)
    link=defaultdict(dict)
    references=defaultdict(dict)

    distinctWords=[]

# tqdm is used to show the progress box. Just there for aesthetic purposes
    for key in tqdm(sorted(data.keys())):
        documents=data[key]
        for i in range(len(documents)):
            posting=documents[i]
            documentID = re.sub(r'.*d([0-9]*).*', r'\1', posting)
            temp = re.sub(r'.*c([0-9]*).*', r'\1', posting)
            if len(temp)>0 and posting!=temp:
                category[key][documentID] = float(temp)
            
            temp = re.sub(r'.*i([0-9]*).*', r'\1', posting)
            if len(temp)>0 and posting != temp:
                info[key][documentID] = float(temp)
            
            temp = re.sub(r'.*l([0-9]*).*', r'\1', posting)
            if len(temp)>0 and posting != temp:
                link[key][documentID] = float(temp)
            
            temp = re.sub(r'.*b([0-9]*).*', r'\1', posting)
            if len(temp)>0 and posting != temp:
                body[key][documentID] = float(temp)
            
            temp = re.sub(r'.*t([0-9]*).*', r'\1', posting)
            if len(temp)>0 and posting != temp:
                title[key][documentID] = float(temp)
            
            temp = re.sub(r'.*r([0-9]*).*', r'\1', posting)
            if len(temp)>0 and posting != temp:
                references[key][documentID] = float(temp)
            
            
        string = key+' '+str(finalCount)+' '+str(len(documents))
        offset.append(str(offsetSize))
        offsetSize+=len(string)+1
        distinctWords.append(string)
    

    titleData=[]
    titleOffset=[]
    prevTitle=0

    bodyData = []
    bodyOffset = []
    prevBody = 0

    infoData = []
    infoOffset = []
    prevInfo = 0

    categoryData = []
    categoryOffset = []
    prevCategory = 0
    
    linkData = []
    linkOffset = []
    prevLink = 0

    referencesData=[]
    referencesOffset=[]
    prevReferences=0

    for key in tqdm(sorted(data.keys())):
        if key in title:
            docs=title[key]
            docs=sorted(docs,key= docs.get, reverse= True)
            string=key+' '
            for doc in docs:
                string+=doc+' '+str(title[key][doc])+' '
            titleData.append(string)
            titleOffset.append(str(prevTitle) + ' ' + str(len(docs)))
            prevTitle += len(string) + 1
        
        if key in body:
            docs = body[key]
            docs = sorted(docs, key = docs.get, reverse=True)
            string = key + ' '
            for doc in docs:
                string += doc + ' ' + str(body[key][doc]) + ' '
            bodyData.append(string)
            bodyOffset.append(str(prevBody) + ' ' + str(len(docs)))
            prevBody += len(string) + 1
        
        if key in info:
            docs = info[key]
            docs = sorted(docs, key = docs.get, reverse=True)
            string = key + ' '
            for doc in docs:
                string += doc + ' ' + str(info[key][doc]) + ' '
            infoData.append(string)
            infoOffset.append(str(prevInfo) + ' ' + str(len(docs)))
            prevInfo += len(string) + 1

        if key in category:
            docs = category[key]
            docs = sorted(docs, key = docs.get, reverse=True)
            string = key + ' '
            for doc in docs:
                string += doc + ' ' + str(category[key][doc]) + ' '
            categoryData.append(string)
            categoryOffset.append(str(prevCategory) + ' ' + str(len(docs)))
            prevCategory += len(string) + 1
        

        if key in link:
            docs = link[key]
            docs = sorted(docs, key = docs.get, reverse=True)
            string = key + ' '
            for doc in docs:
                string += doc + ' ' + str(link[key][doc]) + ' '
            linkData.append(string)
            linkOffset.append(str(prevLink) + ' ' + str(len(docs)))
            prevLink = prevLink + len(string) + 1
        

        if key in references:
            docs = references[key]
            docs = sorted(docs, key = docs.get, reverse=True)
            string = key + ' '
            for doc in docs:
                string += doc + ' ' + str(references[key][doc]) + ' '

            referencesData.append(string)
            referencesOffset.append(str(prevReferences) + ' ' + str(len(docs)))
            prevReferences += len(string) + 1
        

    thread=[]
    thread.append(writeThread('t', titleData, titleOffset, final_count))
    thread.append(writeThread('b', bodyData, bodyOffset, final_count))
    thread.append(writeThread('i', infoData, infoOffset, final_count))
    thread.append(writeThread('c', categoryData, categoryOffset, final_count))
    thread.append(writeThread('l', linkData, linkOffset, final_count))
    thread.append(writeThread('r', referencesData, referencesOffset, final_count))

    i=0
    total=6
    while(i<total):
        thread[i].start()
        i+=1
    i=0
    while(i<total):
        thread[i].join()
        i+=1
    file_name = './files/offset.txt' 
    with open(file_name, 'a') as f:
        f.write('\n'.join(offset))
        f.write('\n')
    file_name = './files/vocab.txt' 
    with open(file_name, 'a') as f:
        f.write('\n'.join(distinctWords))
        f.write('\n')
    return offsetSize ,finalCount+1

        




            







#*************************************************************************************#

#**************************************************************************************#
# CREATING A DICTIONARY OF KEYS AS TITLE,BODY,INFO etc. TOKENS AND 
# THEIR FREQUENCIES AS DICTIONARY VALUES

def creating_dictionary(title, body, info, categories,links,references):
    words=defaultdict(int)
    title_dict=defaultdict(int)
    try:
        for word in title:
            title_dict[word]+=1
            words[word]+=1
    except:
        pass
    
    
    body_dict=defaultdict(int)
    try:
        for word in body:
            body_dict[word]+=1
            words[word]+=1
    except:
        pass
    
    info_dict=defaultdict(int)
    try:
        for word in info:
            info_dict[word]+=1
            words[word]+=1
    except:
        pass
    
    categories_dict=defaultdict(int)
    try:
        for word in categories:
            categories_dict[word]+=1
            words[word]+=1
    except:
        pass

    links_dict=defaultdict(int)
    try:
        for word in links:
            links_dict[word]+=1
            words[word]+=1
    except:
        pass
    
    references_dict=defaultdict(int)
    try:
        for word in references:
            references_dict[word]+=1
            words[word]+=1
    except:
        pass
    
    
    return title_dict, body_dict, info_dict, categories_dict,links_dict,references_dict,words



def creating_inverted_index(title_dict, body_dict, info_dict, categories_dict,links_dict,references_dict,words):
    global pages,files,inverted_index,offset,dictionary
    ID=pages
   
  # posting format is id followed by page number with the delimiters t,b,l,i,r,c followed by the frequency of the word
    
    for word in words.keys():
        string = 'id'+str(ID)
         # t is the delimiter for titles
        if title_dict[word]>0:
            string += 't' + str(title_dict[word])
            

        
        # b is the delimiter for body
        if body_dict[word]>0:
            string += 'b' + str(body_dict[word])
            
        #i is the delimiter for body
        if info_dict[word]>0:
            string += 'i' + str(info_dict[word])
            
        
        # c is the delimiter for categories
        if categories_dict[word]>0:
            string += 'c' + str(categories_dict[word])
            

        # l is the delimiter for links
        if links_dict[word]>0:
            string += 'l' + str(links_dict[word])
           

        if references_dict[word]>0:
            string += 'r' + str(references_dict[word])
            
        inverted_index[word].append(string)
    pages=pages+1
    #print(pages)

    # This is to ensure that every file will have  20000 pages or less
    if pages%20000 == 0:
            offset = writeIntoFile(inverted_index, files,dictionary,offset)
            inverted_index = defaultdict(list)
            dictionary = {}
            files += 1














#*******************************************************************************************************************#
 #******************************************Merge Operation***********************************************************#

def merge(files):
    #This flags list is used to signify if the words in the file have been added into heap
    flags=[0]*files 

    # Below is the heap array
    heap=[]

    a_line_of_inverted_index={}

    finalCount=0
    offsetSize=0
    file_pointers={}
    line={}
    data=defaultdict(list)
    

    # Pushing all the words in the inverted_indexes of all the files into min-Heap
    for i in range(files):
        f_name =  './files/inverted_index' + str(i) + '.txt'
        file_pointers[i] = open(f_name, 'r')
        # The first line of the file is now stored in the first_line dictionary
        line[i]=file_pointers[i].readline().strip()
        a_line_of_inverted_index[i]=line[i].split()

        # a_line_of_inverted_index[i][0] contains the word and a_line_of_inverted_index[i][1] contains the postings
        word=a_line_of_inverted_index[i][0]

        if word not in heap:
            heapq.heappush(heap,word)
        flags[i]=1
    
    
    count=0

# To check if any line is not read in any file
    while any(flags) == 1:
        # Popping the lexicographically smallest word from the heap
        temp_word=heapq.heappop(heap)
        count+=1

        # When the count of the words reach 100000 we are writing the data to file
        if count%100000 == 0:
            prevFileCount=finalCount
            offsetSize,finalCount=finalWrite(data,finalCount,offsetSize)
            if finalCount != prevFileCount:
                data=defaultdict(list)
            
        
        for i in range(files):
            if flags[i]:
                if temp_word == a_line_of_inverted_index[i][0]:
                    line[i]=file_pointers[i].readline().strip()
                    data[temp_word].extend(line[i][1:])

                    if line[i]!='':
                        a_line_of_inverted_index[i]=line[i].split()
                        if a_line_of_inverted_index[i][0] not in heap:
                            heapq.heappush(heap,a_line_of_inverted_index[i][0])
                        else:
                            flags[i]=0
                            file_pointers[i].close()
    
    offsetSize,finalCount=finalWrite(data,finalCount,offsetSize)

                








    












#******************************************SAX Parser  Module ********************************************************#
class SAXHandler( xml.sax.ContentHandler ):
    flag=0
    def __init__(self):
        self.title = ""
        self.ID = ""
        self.text = ""
        self.curr=""
    
    # Call when an element starts
    def startElement(self, tag, attributes):
        self.curr=tag
    
    # Call when a character is read
    def characters(self, content):
        if self.curr == "id" and SAXHandler.flag==0:
            self.ID = content
            SAXHandler.flag=1

        elif self.curr == "text":
            self.text =self.text+content
        elif self.curr=="title":
            self.title=self.title+content


    # Call when an elements ends
    def endElement(self, tag):
        if(tag=="page"):
            dictionary[pages]=self.title.strip().encode("ascii", errors="ignore").decode()

            title=extractTitle(self.title)

            temp_text = self.text.lower() #Case Folding
            temp_text_split=temp_text.split('==references==')
            if len(temp_text_split) == 1:
                temp_text_split=temp_text.split('== references == ')

            categories=[]
            links=[]
            references=[]
            if(len(temp_text_split)>1):
                categories=extractCategories(temp_text_split[1])
                links=extractLinks(temp_text_split[1])
                references=extractReferences(temp_text_split[1])

            body=extractBody(temp_text_split[0])
            info=extractInfo(temp_text_split[0])
            title_dict, body_dict, info_dict, categories_dict,links_dict,references_dict,words=creating_dictionary(title, body, info, categories,links,references)
            creating_inverted_index(title_dict, body_dict, info_dict, categories_dict,links_dict,references_dict,words)
            
            

            SAXHandler.flag=0
            self.curr=""
            self.title = ""
            self.text = ""
            self.ID = ""





if ( __name__ == "__main__"):

    start = timeit.default_timer()
    #Using SAX Parser to parse XML data...
    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # override the default ContextHandler
    Handler = SAXHandler()
    parser.setContentHandler( Handler )

    #    parser.parse(sys.argv[1])
    for file in os.listdir("./Folder"):
        # print(file)
        try:
            parser.parse("./Folder/"+file)
        except:
            pass
    
    try:
        with open('./files/fileNumbers.txt', 'w') as f:
            f.write(str(pages))
    except:
        os.mkdir('files')
        with open('./files/fileNumbers.txt', 'w') as f:
            f.write(str(pages))
    offset = writeIntoFile(inverted_index, files,dictionary,offset)
    inverted_index = defaultdict(list)
    dictionary = {}
    files = files+1
    merge(files)
    stop = timeit.default_timer()
    print (stop - start)


