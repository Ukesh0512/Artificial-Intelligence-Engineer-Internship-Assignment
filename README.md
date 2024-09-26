# Artificial-Intelligence-Engineer-Internship-Assignment

Artificial Intelligence Engineer 

Internship Assignment

#Step 1: Set up the project structure and install required libraries
#Create a new directory for the project and create the following subdirectories: app, data, and reports. The app directory will contain the Python code, data will store the generated responses, and reports will store the generated reports.
pip install transformers torch torchtext nltk matplotlib

#Step 2: Set up the Hugging Face API and GPT-3 model
#Create a new file app/config.py and add the following code:
import os

HF_API_KEY = "YOUR_HF_API_KEY"
HF_API_SECRET = "YOUR_HF_API_SECRET"

GPT3_MODEL = "gpt3"
#Replace YOUR_HF_API_KEY and YOUR_HF_API_SECRET with your actual Hugging Face API key and secret. 

#Step 3: Develop the user interface and generate responses using GPT-3
#Create a new file app/app.py and add the following code:
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import HF_API_KEY, HF_API_SECRET, GPT3_MODEL

# Load the GPT-3 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(GPT3_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(GPT3_MODEL)

def generate_response(user_input):
    # Tokenize the user input
    inputs = tokenizer.encode_plus(
        user_input,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Generate a response using the GPT-3 model
    output = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

def interact_with_user():
    print("Welcome to the conversation AI!")
    while True:
        user_input = input("You: ")
        response = generate_response(user_input)
        print("AI: ", response)

if __name__ == "__main__":
    interact_with_user()
#This code sets up a simple user interface that prompts the user for input and generates a response using the GPT-3 model.  

#Step 4: Perform sentiment analysis on the generated responses
#Create a new file app/sentiment_analysis.py and add the following code:
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(response):
    sentiment = sia.polarity_scores(response)
    if sentiment['compound'] >= 0.05:
        return "positive"
    elif sentiment['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

def analyze_responses(responses):
    sentiments = []
    for response in responses:
        sentiment = analyze_sentiment(response)
        sentiments.append(sentiment)
    return sentiments   
#This code uses the NLTK library to perform sentiment analysis on the generated responses.

#Step 5: Generate reports and data visualizations
#Create a new file app/report_generation.py and add the following code:
import matplotlib.pyplot as plt

def generate_report(sentiments):
    # Generate a pie chart to visualize the sentiment distribution
    plt.pie(sentiments.count("positive"), sentiments.count("negative"), sentiments.count("neutral"), labels=["Positive", "Negative", "Neutral"], autopct='%1.1f%%')
    plt.title("Sentiment Distribution")
    plt.savefig("reports/sentiment_distribution.png")

    # Generate a report summarizing the sentiment analysis results
    with open("reports/report.txt", "w") as f:
        f.write("Sentiment Analysis Report\n")
        f.write("----------------------------\n")
        f.write("Positive Sentiments: {}\n".format(sentiments.count("positive")))
        f.write("Negative Sentiments: {}\n".format(sentiments.count("negative")))
        f.write("Neutral Sentiments: {}\n".format(sentiments.count("neutral")))

if __name__ == "__main__":
    responses = ["This is a great product!", "I hate this product.", "This product is okay."]
    sentiments = analyze_responses(responses)
    generate_report(sentiments)  
#This code generates a pie chart to visualize the sentiment distribution and a report summarizing the sentiment analysis results.

#Step 6: Integrate the components and deploy the application
#Create a new file app/main.py and add the following code:
import os
from app import app, sentiment_analysis, report_generation

def main():
    # Interact with the user and generate responses
    responses = []
    interact_with_user()
    responses = [response for response in responses if response]

    # Perform sentiment analysis on the generated responses
    sentiments = sentiment_analysis.analyze_responses(responses)

    # Generate reports and data visualizations
    report_generation.generate_report(sentiments)

if __name__ == "__main__":
    main()    
#This code integrates the user interface, response generation, sentiment analysis, and report generation components.

#Step 7: Set up the PostgreSQL database
#Create a new file app/db.py and add the following code:
import psycopg2

def connect_to_db():
    conn = psycopg2.connect(
        dbname="your_database",
        user="your_username",
        password="your_password",
        host="your_host",
        port="your_port"
    )
    return conn

def create_table(conn):
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS responses (id SERIAL PRIMARY KEY, response TEXT, sentiment TEXT)")
    conn.commit()

def insert_response(conn, response, sentiment):
    cur = conn.cursor()
    cur.execute("INSERT INTO responses (response, sentiment) VALUES (%s, %s)", (response, sentiment))
    conn.commit()

def get_responses(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM responses")
    return cur.fetchall()
#This code sets up a PostgreSQL database connection and creates a table to store the generated responses and their corresponding sentiments.

#Step 8: Integrate the database with the application
#Modify the app/main.py file to integrate the database with the application:
import os
from app import app, sentiment_analysis, report_generation, db

def main():
    # Interact with the user and generate responses
    responses = []
    conn = db.connect_to_db()
    db.create_table(conn)

    interact_with_user()
    responses = [response for response in responses if response]

    # Perform sentiment analysis on the generated responses
    sentiments = sentiment_analysis.analyze_responses(responses)

    # Insert responses and sentiments into the database
    for response, sentiment in zip(responses, sentiments):
        db.insert_response(conn, response, sentiment)

    # Generate reports and data visualizations
    report_generation.generate_report(sentiments)

if __name__ == "__main__":
    main()  
#This code integrates the database with the application, storing the generated responses and their corresponding sentiments in the database.

#Step 9: Deploy the application using Docker
#Create a new file Dockerfile and add the following code:
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app/main.py"]   
#This code creates a Docker image for the application. 
