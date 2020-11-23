
# Wikipedia-Search-Engine
This repository consists of a search engine over the 63GB Wikipedia dump. The code consists of indexer.py and search.py. Both simple and multi field queries have been implemented. The search returns a ranked list of articles in real time.

# Indexing:
Parsing: SAX Parser is used to parse the XML corpus.
Casefolding: Converting Upper Case to Lower Case.
Tokenisation: It is done using regex.
Stop Word Removal: Stop words are removed by referring to the stop word list returned by nltk.
Stemming: A python library PyStemmer is used for this purpose.
Creating various index files with word to field postings.
Multi-way External sorting on the index files to create field based files along with their respective offsets.

# Searching:
The query given is parsed, processed and given to the respective query handler(simple or field).
One by one word is searched in vocabulary and the file number is noted.
The respective field files are opened and the document ids along with the frequencies are noted.
The documents are ranked on the basis of TF-IDF scores.
The title of the documents are extracted using title.txt

# Files Produced
index*.txt (intermediate files) : It consists of words with their posting list. Eg. d1b2t4c5 d5b3t6l1
title.txt : It consist of id-title mapping.
titleOffset.txt : Offset for title.txt
vocab.txt : It has all the words and the file number in which those words can be found along with the document frequency.
offset.txt : Offset for vocab.txt
[b|t|i|r|l|c]*.txt : It consists of words found in various sections of the article along with document id and frequency.
offset_[b|t|i|r|l|c]*.txt : Offset for various field files.

# How to run:
Run python3 indexer.py ./path of xml dump ./path of output(Inverted Index)
Now, run python3 search.py , wait for files to be loaded
Now, input queries .
For field queries give space separated in form of t:(title) b:(body) i:(infobox) c:category l:(links)
Output shows top10 relevant documents.


# Details about parsers:
https://www.edureka.co/blog/parsing-xml-file-using-sax-parser/#:~:text=SAX%20Parser%20%E2%80%93%20SAX%20is%20an,acronym%20for%20Document%20Object%20Model.


# Detailed information about Tf-Idf:

How to Implement a Search Engine Part 3: Ranking tf-idf
Posted on July 17, 2011 by Arden
Overview
We have come to the third part of our implementing a search engine project, ranking. The first part was about creating the index, and the second part was querying the index. We basically have a search engine that can answer search queries on a given corpus, but the results are not ranked. Now, we will include ranking to obtain an ordered list of results, which is one of the most challenging and interesting parts. The first ranking scheme we will implement is tf-idf. In the following articles, we’ll analyze Okapi BM25, which is a variant of tf-idf. We will also implement Google’s PageRank. Then we will explore Machine Learning techniques such as Naive Bayes Classifier, Support Vector Machines (SVM), Clustering, Decision Trees and so forth.

Tf-idf is a weighting scheme that assigns each term in a document a weight based on its term frequency (tf) and inverse document frequency (idf).  The terms with higher weight scores are considered to be more important. It’s one of the most popular weighting schemes in Information Retrieval.


Term Frequency – tf
Let’s first define how term frequency is calculated for a term t in document d. It is basically the number of occurrences of the term in the document.

$latex
tf_{t,d} = N_{t,d}
&s=2$

We can see that as a term appears more in the document it becomes more important, which is logical. However, there is a drawback, by using term frequencies we lose positional information. The ordering of terms doesn’t matter, instead the number of occurrences becomes important. This is known as the bag of words model, and it is widely used in document classification. In bag of words model, the document is represented as an unordered collection of words. However, it doesn’t turn to be a big loss. Of course we lose the semantic difference between “Bob likes Alice” and “Alice likes Bob”, but we still get the general idea.

We can use a vector to represent the document in bag of words model, since the ordering of terms is not important. There is an entry for each unique term in the document with the value being its term frequency. For the sake of an example, consider the document “computer study computer science”. The vector representation of this document will be of size 3 with values [2, 1, 1] corresponding to computer, study, and science respectively. We can indeed represent every document in the corpus as a k-dimensonal vector, where k is the number of unique terms in that document. Each dimension corresponds to a separate term in the document. Now every document lies in a common vector space. The dimensionality of the vector space is the total number of unique terms in the corpus. We will further analyze this model in the following sections. The representation of documents as vectors in a common vector space is known as the vector space model and it’s very fundamental to information retrieval. It was introduced by Gerard Salton, a pioneer of information retrieval. Google’s core ranking team is led by Amit Singhal, who was the PhD student of Salton at Cornell University.

While using term frequencies if we use pure occurrence counts, longer documents will be favored more. Consider two documents with exactly the same content but one being twice longer by concatenating with itself.  The tf weights of each word in the longer document will be twice the shorter one, although they essentially have the same content. To remedy this effect, we length normalize term frequencies. So, the term frequency of a term t in document D now becomes:

$latex
tf_{t,d} = \dfrac{N_{t,d}}{||D||}
&s=2$

||D|| is known as the Euclidean norm and is calculated by taking the square of each value in the document vector, summing them up, and taking the square root of the sum. After normalizing the document vector, the entries are the final term frequencies of the corresponding terms. The document vector is also a unit vector, having a length of 1 in the vector space.

Inverse Document Frequency – idf
We can’t only use term frequencies to calculate the weight of a term in the document, because tf considers all terms equally important. However, some terms occur more rarely and they are more discriminative than others. Suppose we search for articles about computer vision. Here the term vision gives us more information about the intent of the query, instead of the term computer. We don’t simply want articles that are about computers, we want them to be about vision. If we purely use tf values then the term computer will dominate because it’s a more common term than vision, and the articles containing computer will be ranked higher. To mitigate this effect, we use inverse document frequency. Let’s first see what document frequency is. The document frequency of a term t is the number of documents containing the term:

$latex
df_t = N_t
&s=2$

Note that the occurrence counts of the term in the individual documents is not important. We are only interested in whether the term is present in a document or not, without taking into consideration the counts. It’s like a binary 0/1 counting. If we were to consider the number of occurrences in the documents, then it’s called collection frequency. But document frequency proves to be more accurate. Also note that term frequency is a document-wise statistic while document frequency is collection-wise. Term frequency is the occurrence count of a term in one particular document only; while document frequency is the number of different documents the term appears in, so it depends on the whole corpus. Now let’s look at the definition of inverse document frequency. The idf of a term is the number of documents in the corpus divided by the document frequency of a term. Let’s say we have N documents in the corpus, then the inverse document frequency of term t is:

$latex
idf_t = \dfrac{N}{df_t} = \dfrac{N}{N_t}
&s=2$

This is a very useful statistic, but it also requires a slight modification. Consider a corpus with 1000 documents. A term appears in 10 documents and another term appears in 100, so the document frequencies are 10 and 100 respectively. The inverse document frequencies are 100 and 10. Idf is 100 for the term that has a df of 10 (1000/10), and idf is 10 for the document with df 100 (1000/100) by definition. Now as we can see the term that appears in 10 times more documents is considered to be 10 times less important. It’s expected that the more frequent term to be considered less important, but the factor 10 seems too harsh. Therefore, we take the logarithm of the inverse document frequencies. Let’s say the base of log is 2, than term that appears 10 times less often is considered to be around 3 times more important. So, the idf of a term t becomes:

$latex
idf_t = log\dfrac{N}{df_t}
&s=2$

This is better, and since log is a monotonically increasing function we can safely use it. Notice that idf never becomes negative because the denominator (df of a term) is always less than or equal to the size of the corpus (N). When a term appears in all documents, its df = N, then its idf becomes log(1) = 0. Which is ok because if a term appears in all documents, it doesn’t help us to distinguish between them. It’s basically a stopword, such as “the”, “a”, “an” etc. Also notice the resemblance between idf and the definition of entropy in information theory. In our case p(x) is df/N, which is the probability of seeing a term in a randomly chosen document. And idf is –logp(x). The important result to note is, as more rare events occur, the information gain increases. Which means less frequent terms gives us more information.

Tf-idf scoring
We have defined both tf and idf, and now we can combine these to produce the ultimate score of a term t in document d. We will again represent the document as a vector, with each entry being the tf-idf weight of the corresponding term in the document. The tf-idf weight of a term t in document d is simply the multiplication of its tf by its idf:

$latex
tf\mbox{-}idf_{t,d} = tf_{t,d} \cdot idf_t
&s=2$

Let’s say we have a corpus containing K unique terms, and a document containing k unique terms. Using the vector space model, our document becomes a k-dimensional vector in a K-dimensional vector space. Generally k will be much less than K, because all terms in the corpus won’t appear in a single document. The values in the vector corresponding to the k terms that appear in the document will be their respective tf-idf weights, computed by the formula above. The entries corresponding to the K-k terms that don’t appear in the current document will be 0. Because their tf weight in the current document will be 0, since they don’t occur. Note that their idf scores won’t be 0, because idf is a collection-wise statistic, which depends on all the documents in the corpus. But tf is a document-wise statistic, which only depends on the current document. So, if a term doesn’t appear in the current document, it gets a tf score of 0. Multiplying tf and idf, the tf-idf weights of the missing K-k terms become 0. So, in the end we have a sparse vector with most of the entries being 0. To sum everything up, we represent documents as vectors in the vector space. A document vector has an entry for every term, with the value being its tf-idf score in the document.

We will also represent the query as a vector in the same K-dimensional vector space. It will have much fewer dimensions though, since queries are generally much shorter than the documents. Now let’s see how to find relevant documents to a query. Since both the query and the documents are represented as vectors in a common vector space, we can take advantage of this. We will compute the similarity score between the query vector and all the document vectors, and select the ones with top similarity values as relevant documents to the query. Before computing the similarity scores between vectors, we will perform one final operation as we did before, normalization. We will normalize both the query vector and all the document vectors, obtaining unit vectors.

Now that we have everything we need, we can finally compute the similarity scores between the query and document vectors, and rank the documents. The similarity score between two vectors in a vector space is the the angle between them. If two documents are similar they will be close to each other in the vector space, having a small angle in between. So given the vector representation of the documents, how do we compute the angle between them? We can do it very easily if the vectors are already normalized, which is true in our case, and this technique is called cosine similarity. We take the dot product of the vectors and the result is the cosine value of the angle between them. Remember that when the angle is smaller its cosine value is larger, so when two vectors are similar their cosine similarity value will be larger. This gives us a great similarity metric with higher values meaning more similar and lower values meaning less. Therefore, if we compute the cosine similarity between the query vector and all the document vectors, sort them in descending order, and select the documents with top similarity, we will obtain an ordered list of relevant documents to this query. Voila! We now have a systematic methodology to get an ordered list of results to a query, ranking.
