import sacrebleu

filename_list = ['2_2/', '4_2/', '4_4/', 'baseline/']

score_list = []
self_bleu = {}

for identify in filename_list:
    filename = './output/'+identify+'generated_predictions.txt'
    with open(filename, 'r') as fp:
        sen_list = fp.readlines() #all sentences
        num = len(sen_list) #number of sentences
        for i in range(num):
            refs = [[sen_list[j].replace('\n', '').replace(' ','　')] for j in range(num) if j!=i]
            sys = [sen_list[i].replace('\n', '').replace(' ','　')]  
            blue = sacrebleu.corpus_bleu(sys, refs) #calcuate score
            score_list.append(blue.score) #save the score
            #show each score (Comment out if not needed)
            # print(num, sum(score_list) / num)

    #average score
    print(sum(score_list) / num)
    self_bleu[identify] = sum(score_list) / num

print(self_bleu)