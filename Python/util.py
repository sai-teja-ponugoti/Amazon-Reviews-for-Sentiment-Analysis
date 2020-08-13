

# function to load data from data directry
def get_labels_and_texts(file):
    count = 0
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        count += 1
        x = line.decode("utf-8")
        # appending the labels
        labels.append(int(x[9]) - 1)
        #appending the text to texts
        texts.append(x[10:].strip())
        # counter to show the progress
        if count%400000 == 0:
            print("done with files:",count)
    return texts, np.array(labels)


# function to reprocess texts by removing special charectrs and non ascii charecters
def normalize_texts(texts):
    count = 0
    normalized_texts = []
    for text in texts:
        count += 1
        lower = text.lower()
        # removing non alphanumeric chars
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        # removing non ascii chars
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
        # counter to indicate progress
        if count%200000 == 0:
            print("done :",count)
    return normalized_texts



# function to split training data to train and validation sets
def train_text_split_function(data, labels, size_text):
    return train_test_split(data, labels, random_state= 42, test_size= size_text )


# function to pad tokenized sequences to constant length
def padding_seq(data):
    return tensorflow.keras.preprocessing.sequence.pad_sequences(data, maxlen=maxlen, padding = 'post', truncating='post')