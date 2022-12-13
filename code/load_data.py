import torch
import jsonlines
import pandas as pd
from collections import Counter, defaultdict
import torch.nn.functional as F
from datasets import Dataset, DatasetDict

class SQuADDataset(Dataset):
    def __init__(self, passage_list, question_list, answer_list, tokenizer, src_max_length, tgt_max_length):

        '''
        src_max_length: data_args.max_source_length
        '''
        self.source_input_ids = []
        self.source_masks = []
        self.target_input_ids = []
        self.target_masks = []

        for passage, question, answer in zip(passage_list, question_list, answer_list):
            
            answer_text = answer
            len_ans = len(answer)
            ans_start = passage.index(answer_text)
            source_text = f'问题生成: {passage[:ans_start]}[HL]{answer_text}[HL]{passage[len_ans + ans_start:]}'

            source_text_encodings_dict = tokenizer(source_text, truncation=True, max_length=src_max_length, padding="max_length")

            target_text = question
            target_text_encodings_dict = tokenizer(target_text, truncation=True, max_length=tgt_max_length, padding="max_length")

            ##askaskask
            source_ids = source_text_encodings_dict["input_ids"]
            source_mask = source_text_encodings_dict["attention_mask"]
            target_ids = target_text_encodings_dict["input_ids"]
            target_mask = target_text_encodings_dict["attention_mask"]

            self.source_input_ids.append(torch.tensor(source_ids))
            self.source_masks.append(torch.tensor(source_mask))
            self.target_input_ids.append(torch.tensor(target_ids))
            self.target_masks.append(torch.tensor(target_mask))

    def __len__(self):
        return len(self.source_input_ids)

    def __getitem__(self, idx):
        
        return {
            "source_ids": self.source_input_ids[idx],
            "source_mask": self.source_masks[idx],
            "target_ids": self.target_input_ids[idx],
            "target_ids_y":  self.target_input_ids[idx],
        }


def select_answer(answers):
        '''
        We select answers using the following rules:
        1. voting
        2. the shortest one.
        '''
        if len(answers) == 1:
            return answers[0]

        # Vote for the popular answer
        start_pos: dict = defaultdict(list)
        votes: Counter = Counter()
        for ans_dict in answers:
            answer_text = ans_dict["text"]
            ans_char_start_pos = ans_dict["answer_start"]
            start_pos[answer_text].append(ans_char_start_pos)
            votes[answer_text] += 1

        # if we have agreement (i.e. # of votes != 1)
        ans, n_vote = votes.most_common(1)[0]
        if n_vote != 1:
            return {
                "text": ans,
                "answer_start": start_pos[ans][0]
            }

        # if equal votes, select the shortest one
        min_len = 9999
        idx = -1
        for i, ans_dict in enumerate(answers):
            len_ = len(ans_dict["text"])
            if len_ > min_len:
                idx = i
                min_len = len_
        ret = {
            "text": answers[idx]["text"],
            "answer_start": answers[idx]["answer_start"]
        }
        return ret

def load_squad_dataset(path, util_size = None):
    """
    path: string with 1 or more datapaths
    tokenizer: separation tokenizer
    util_size: test the validity of desiigned model with different scale of training dataset

    """
    path_list = path.split(",")  #input multiple datasets

    contexts, questions, answers = [], [], []

    if type(path_list) is list:
        for path in path_list:
            print(f"parsing data in {path}....")
            with jsonlines.open(path, 'r') as reader:
                cnter = 0
                for line in reader:
                    cnter +=1
                    #print(line)
                    if "context" not in line.keys() or "question" not in line.keys() or "answer" not in  line.keys():
                        print(f"Missing crucial key in line {cnter}, {line.keys()}")
                        continue
                    contexts.append(line["context"])
                    questions.append(line["question"])
                    answers.append(line["answer"])
    else:
        with jsonlines.open(path, 'r') as reader:
            for line in reader:  #dict
                cnter +=1
                if "context" not in line.keys() or "question" not in line.keys() or "answer" not in  line.keys():
                    print(f"Missing crucial key in line {cnter}, {line.keys()}")
                    continue
                contexts.append(line["context"])
                questions.append(line["question"])
                answers.append(line["answer"])
    
    l = len(contexts)
    if util_size is not None:
        l_ = int(util_size*l)
        contexts = contexts[:l_]
        questions = questions[:l_]
        answers = answers[:l_]
    
    #train_dataset = SQuADDataset(contexts, questions, answers, tokenizer, util_size, src_max_length=512, tgt_max_length=30)

    return Dataset.from_dict({"context": contexts, "question": questions, "answer": answers})

def my_load_dataset(train_path = None, eval_path = None, test_path = None, util_size = None):
    train_dataset,eval_dataset,test_dataset=  None,None,None
    if train_path is not None:
        train_dataset = load_squad_dataset(train_path, util_size)
    if eval_path is not None:
        eval_dataset = load_squad_dataset(eval_path, util_size)
    if test_path is not None:
        test_dataset = load_squad_dataset(test_path, util_size)
        
    return DatasetDict(
        {
        "train": train_dataset,
        "evaluation":  eval_dataset,
        "test":  test_dataset
        }
    )
