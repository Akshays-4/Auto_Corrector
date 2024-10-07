pip install happytransformer
pip install transformers


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")

def correct_grammar(text):
    inputs = tokenizer.encode("fix: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

b = np.zeros(len(x), dtype=int) 

for i in range(len(x)):
    original_answer = x[i]
    corrected_answer = correct_grammar(original_answer)

    # Compare original and corrected answers
    num_changes = sum(1 for a, b in zip(original_answer.split(), corrected_answer.split()) if a != b)

    # Penalize based on the number of grammar corrections
    b[i] = num_changes




from happytransformer import HappyTextToText

happy_tt = HappyTextToText("T5", "t5-base")

from datasets import load_dataset

train_dataset = load_dataset("jfleg", split='validation[:]')

eval_dataset = load_dataset("jfleg", split='test[:]')

# for case in train_dataset["corrections"][:2]:
#   print(case)
#   print(case[0])
#   print("--------------------------------------------------------")

# preprocessing
import csv

def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["input", "target"])
        for case in dataset:
     	    # Adding the task's prefix to input 
            input_text = "grammar: " + case["sentence"]
            for correction in case["corrections"]:
                # a few of the cases contain blank strings. 
                if input_text and correction:
                    writter.writerow([input_text, correction])
                    


generate_csv("train.csv", train_dataset)
generate_csv("eval.csv", eval_dataset)

before_result = happy_tt.eval("eval.csv")
print("Before loss:", before_result.loss)

from happytransformer import TTTrainArgs

args = TTTrainArgs(batch_size=8)
happy_tt.train("train.csv", args=args)

before_loss = happy_tt.eval("eval.csv")

print("After loss: ", before_loss.loss)


from happytransformer import TTSettings
beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=20)
example_1 = "grammar: This sentences, has bads grammar and spelling!"
result_1 = happy_tt.generate_text(example_1, args=beam_settings)
print(result_1.text)
# Result: This sentence has bad grammar and spelling!