import os
import json
import pandas as pd
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch

# Load T5 model and tokenizer
model_name = "google/flan-t5-base"  # You can try a larger model if needed
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Create a T5 pipeline for text2text-generation
gen_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=(0 if torch.cuda.is_available() else -1),
    max_length=200,  # Lower max_length per question
    temperature=0.7,
    do_sample=True,
)

# Load and truncate input text
file_path = r"E:\ML\gen AI\mcqgen\data.txt"
with open(file_path, "r") as file:
    TEXT = file.read()

max_input_tokens = 250  # adjust if needed
input_ids = tokenizer.encode(TEXT, truncation=True, max_length=max_input_tokens)
truncated_text = tokenizer.decode(input_ids, skip_special_tokens=True)

# We'll generate 5 questions one by one.
NUM_QUESTIONS = 5
rows = []

# Define a prompt template for generating a single MCQ.
# We instruct the model to output a single line in a fixed format.
single_q_prompt = """
You are an expert MCQ generator.
Given the following text, generate one multiple-choice question for biology students in a simple tone.
Output exactly one line in the following format (do not include any extra text):

MCQ: <question text> | Choices: a: <option A> | b: <option B> | c: <option C> | d: <option D> | Correct: <letter>

Text: {text}

Example:
MCQ: What is the scientific study of life called? | Choices: a: Physics | b: Chemistry | c: Biology | d: Geology | Correct: c
"""

# Loop to generate each question individually
for i in range(NUM_QUESTIONS):
    prompt = single_q_prompt.format(text=truncated_text)
    output = gen_pipeline(prompt)[0]["generated_text"]
    print(f"Raw output for question {i+1}:", output)

    # Use a regex to extract the line starting with "MCQ:" and containing the expected delimiters.
    match = re.search(
        r"MCQ:\s*(.*?)\s*\|\s*Choices:\s*(.*?)\s*\|\s*Correct:\s*([a-dA-D])", output
    )
    if match:
        mcq_text = match.group(1).strip()
        choices_text = match.group(2).strip()
        correct_letter = match.group(3).lower().strip()
        # You can perform additional cleanup on choices_text if needed.
        rows.append(
            {"MCQ": mcq_text, "Choices": choices_text, "Correct": correct_letter}
        )
    else:
        print(f"Failed to parse output for question {i+1}, using placeholder.")
        rows.append(
            {
                "MCQ": "Placeholder question",
                "Choices": "a: Option A | b: Option B | c: Option C | d: Option D",
                "Correct": "a",
            }
        )

# Create a DataFrame and save to CSV
quiz_df = pd.DataFrame(rows)
quiz_df.to_csv("biology_quiz.csv", index=False)
print("Quiz CSV saved as biology_quiz.csv")


'''
### code using GPT-j to generate multiple choice questions but it is not working
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_huggingface import HuggingFacePipeline  # Updated import
import torch

# Load GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", offload_folder="offload"
)

# Create a LangChain-compatible pipeline for GPT-J without specifying device here
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=500,  # Control the output length
    temperature=0.7,
    do_sample=True,
)

# Wrap the pipeline in a LangChain LLM
llm = HuggingFacePipeline(pipeline=generator)

# Define the response JSON structure
RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

# Define the prompt template for quiz generation
TEMPLATE = """
You are an expert MCQ generator. Given the following text, generate {number} multiple-choice questions for {subject} students in a {tone} tone.
Ensure the format matches the JSON structure below.

Text: {text}

### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE,
)

# Create the quiz generation chain
quiz_chain = LLMChain(
    llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True
)

# Define the quiz evaluation prompt
TEMPLATE2 = """
You are an expert in educational content. Given the following quiz, evaluate its complexity and provide an analysis (max 50 words).
If necessary, update the questions to fit the students' level.

Quiz:
{quiz}
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["quiz"],
    template=TEMPLATE2,
)

review_chain = LLMChain(
    llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True
)

generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True,
)

# Load input text
file_path = "data.txt"
with open(file_path, "r") as file:
    TEXT = file.read()

# Truncate text to fit within model's limit
max_input_tokens = 300
input_ids = tokenizer.encode(TEXT, truncation=True, max_length=max_input_tokens)
truncated_text = tokenizer.decode(input_ids, skip_special_tokens=True)

# Set quiz parameters
NUMBER = 5
SUBJECT = "biology"
TONE = "simple"

# Generate and evaluate the quiz
response = generate_evaluate_chain.invoke(
    {
        "text": truncated_text,
        "number": NUMBER,
        "subject": SUBJECT,
        "tone": TONE,
        "response_json": json.dumps(RESPONSE_JSON),
    }
)

# Extract and parse quiz JSON
try:
    quiz_output = response["quiz"]
    start_index = quiz_output.find("{")
    end_index = quiz_output.rfind("}") + 1
    quiz_json = quiz_output[start_index:end_index]
    quiz = json.loads(quiz_json)
except (json.JSONDecodeError, ValueError):
    print("Failed to parse quiz output. Using fallback.")
    quiz = RESPONSE_JSON

# Ensure quiz contains the required number of questions
while len(quiz) < NUMBER:
    quiz[str(len(quiz) + 1)] = {
        "mcq": "Placeholder question",
        "options": {"a": "Option A", "b": "Option B", "c": "Option C", "d": "Option D"},
        "correct": "a",
    }

# Save quiz to CSV
quiz_df = pd.DataFrame(
    [
        {
            "MCQ": q["mcq"],
            "Choices": " | ".join(f"{k}: {v}" for k, v in q["options"].items()),
            "Correct": q["correct"],
        }
        for q in quiz.values()
    ]
)
quiz_df.to_csv("biology_quiz.csv", index=False)

# Print quiz review
print("Quiz Review:", response["review"])
'''
