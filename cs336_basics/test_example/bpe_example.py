vocab = {}
for i in range(256):
    vocab[i] = bytes([i])
vocab[len(vocab)] = b"<|endoftext|>"

print(vocab[256])

#   pre_tokenization
pre_token = {}
corpus = "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
for word in corpus.split():
    word_encode = word.encode('utf-8')
    word_bytes = tuple(bytes([b]) for b in word_encode)
    pre_token[word_bytes] = pre_token.get((word_bytes),0) + 1

print(pre_token)

#merge
merges = []
merge_time = 6

for i in range(merge_time):
    # record the potential pair and the frequncy
    tem_pair = {}           #用来记录可能merge的字节对和对应的频率  
    for word,freq in pre_token.items():
        for j in range(len(word)-1):                 #word 是tuple[bytes,...]
            pair = (word[j],word[j+1])
            tem_pair[pair] = tem_pair.get(pair,0) + freq

    print(tem_pair)
    # find the most frequncy pair
    sorted_tem_pair = sorted(tem_pair.items(),key = lambda x: (x[1],x[0]),reverse = True)       #return a new list
    (a,b),freq = sorted_tem_pair[0]
    merges.append((a,b))
    merged_pair = a + b
    vocab[len(vocab)] = merged_pair

    print(merged_pair)

    # merge
    new_pre_token = {}          #将原来的pre_token重新替换为新的
    for word,freq in pre_token.items():
        new_word = []
        j = 0
        while j < len(word):
            if j < len(word)-1 and word[j]+word[j+1] == merged_pair:
                new_word.append(word[j]+word[j+1])
                j+=2
                continue
            new_word.append(word[j])
            j+=1
        new_pre_token[tuple(new_word)] = freq
    pre_token = new_pre_token

    print(pre_token)
print(merges)
print(vocab[258])