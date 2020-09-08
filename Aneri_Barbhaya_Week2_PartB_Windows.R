##SENTIMENT ANALYSIS USING NAIVE BAYES

#STEP 1- collecting the data
# Dataset from github on the links/headlines which are classified as clickbait and not clickbait. 21936 observations with 2 features  title and label.
#title is the actual title and label provides information on whether the title/link is clickbait or not.

#STEP 2- exploring and preparing the data
#reading the csv
cnc_raw <- read.csv("unlabelled.csv", stringsAsFactors = FALSE)

#examining the structure of the dataset
str(cnc_raw)

#recoding label variable as factor
cnc_raw$label <- factor(cnc_raw$label)

#checking again the structure to ensure label has been recoded properly
str(cnc_raw$label)
table(cnc_raw$label)

#processing title data for analysis
install.packages("tm")
library(tm)

#Creating a collection of title documents using Corpus
cnc_corpus <- VCorpus(VectorSource(cnc_raw$title))
print(cnc_corpus)

#checking the content of corpus
inspect(cnc_corpus[1:2])
as.character(cnc_corpus[[1]])
lapply(cnc_corpus[1:2], as.character)

#tm_map method to transform tm corpus
cnc_corpus_clean <- tm_map(cnc_corpus,content_transformer(tolower))
as.character(cnc_corpus[[1]])
as.character(cnc_corpus_clean[[1]])
cnc_corpus_clean <- tm_map(cnc_corpus_clean, removeNumbers)
cnc_corpus_clean <- tm_map(cnc_corpus_clean,removeWords, stopwords())
cnc_corpus_clean <- tm_map(cnc_corpus_clean, removePunctuation)
removePunctuation("hello...world")

install.packages("SnowballC")
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))
cnc_corpus_clean <- tm_map(cnc_corpus_clean, stemDocument)
cnc_corpus_clean <- tm_map(cnc_corpus_clean, stripWhitespace)

#tokenization - split the messages into individual components - words
#creating sparse matrix
cnc_dtm <- DocumentTermMatrix(cnc_corpus_clean)
cnc_dtm2 <- DocumentTermMatrix(cnc_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

cnc_dtm
cnc_dtm2

#creating training and test datasets (75% training 25% testing)
cnc_dtm_train <- cnc_dtm[1:16452, ]
cnc_dtm_test  <- cnc_dtm[16453:21936, ]

cnc_train_labels <- cnc_raw[1:16452, ]$label
cnc_test_labels  <- cnc_raw[16453:21936, ]$label

prop.table(table(cnc_train_labels))
prop.table(table(cnc_test_labels))


#Visualizing title data-word clouds
install.packages("wordcloud")
library(wordcloud)
wordcloud(cnc_corpus_clean, min.freq = 200, random.order = FALSE)

clickbait<- subset(cnc_raw, label == "clickbait")
not_clickbait<- subset(cnc_raw, label == "not-clickbait")

wordcloud(clickbait$title, max.words = 50, scale = c(3, 0.5))
wordcloud(not_clickbait$title, max.words = 50, scale = c(3, 0.5))

#creating indicator features for frequent words
findFreqTerms(cnc_dtm_train, 5)
cnc_freq_words <- findFreqTerms(cnc_dtm_train, 5)
str(cnc_freq_words)

cnc_dtm_freq_train<- cnc_dtm_train[ , cnc_freq_words]
cnc_dtm_freq_test <- cnc_dtm_test[ , cnc_freq_words]

#coverting count of word represented in cells of sparse matrix to categorical 
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

#applying convert_count function to all the columns
cnc_train <- apply(cnc_dtm_freq_train, MARGIN = 2,convert_counts)
cnc_test <- apply(cnc_dtm_freq_test, MARGIN = 2,convert_counts)

#STEP 3- Training a model on the data
install.packages("e1071")
library(e1071)
#building the classifier
cnc_classifier <- naiveBayes(cnc_train, cnc_train_labels)

#STEP 4- evaluating model performance
#making predictions
cnc_test_pred <- predict(cnc_classifier, cnc_test)

library(gmodels)
CrossTable(cnc_test_pred, cnc_test_labels,prop.chisq = FALSE, prop.t = FALSE,dnn = c('predicted', 'actual'))

#STEP 5- improving model performance
cnc_classifier2 <- naiveBayes(cnc_train, cnc_train_labels,laplace = 1)
cnc_test_pred2 <- predict(cnc_classifier2, cnc_test)
CrossTable(cnc_test_pred2, cnc_test_labels,prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,dnn = c('predicted', 'actual'))
