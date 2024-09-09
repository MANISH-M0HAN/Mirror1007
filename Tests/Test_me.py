import csv
import requests
import os
from dotenv import load_dotenv
from request_response_csv import request_response
from datetime import datetime, timezone, timedelta

load_dotenv()
username = input("Please enter your name: ").upper()
url = os.getenv("CHATBOT_URL")
api_key = os.getenv("API_KEY")

headers = {
    "X-API-KEY": api_key,
    "Content-Type": "application/json"
}

data = [
    ["Concept", "Input", "Expected Output"],
    ["Single trigger", "what is angina?", "Angina is chest pain caused by reduced blood flow to the heart."],
    ["Single trigger", "what is angina? and why is it caused", "Angina is chest pain caused by reduced blood flow to the heart. It signals an increased risk of heart attack."],
    ["Only trigger word", "angina", "Angina is chest pain caused by reduced blood flow to the heart.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
    ["Only trigger word - multiple", "angina cardiovascular", "Cardiovascular refers to the heart and blood vessels. \n\n Angina is chest pain caused by reduced blood flow to the heart.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
    ["Domain relevance", "what is cardiac", "Cardiovascular refers to the heart and blood vessels. \n\n Myocardial infarction is a heart attack. \n\n Tachycardia is a rapid heartbeat."],
    ["Domain relevance", "what is football", "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"],
    ["Fallback", "what health", "This is a placeholder response generated for your question."],
    ["multiple triggers", "what is cholesterol and what is angina", "Cholesterol is a type of fat found in your blood. \n\n Angina is chest pain caused by reduced blood flow to the heart."],
    ["multiple triggers and multiple intent", "what is cholesterol and how is angina caused", "Cholesterol is a type of fat found in your blood. Eat a low-fat diet, exercise regularly, and take prescribed medications. High levels increase the risk of heart disease. \n\n Angina is chest pain caused by reduced blood flow to the heart. Manage stress, avoid heavy meals, and follow a healthy lifestyle. It signals an increased risk of heart attack."],
    ["only intent", "how", "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"],
    ["Spell check", "wat is angina", "Angina is chest pain caused by reduced blood flow to the heart."],
    ["Greetings", "hi", "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"],
    ["Edge Case", "cardiovascular", "Cardiovascular refers to the heart and blood vessels.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
    ["Edge Case", "cardiovascular?", "Cardiovascular refers to the heart and blood vessels.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
    ["Edge Case", "angina cardiovascular?", "Cardiovascular refers to the heart and blood vessels. \n\n Angina is chest pain caused by reduced blood flow to the heart.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
]

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, "test.csv")

def create_csv(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def get_bot_response(user_input):
    try:
        payload = {"user_input": user_input}
        response = requests.post(url, headers=headers, json=payload)
        return response.json().get("response", "")
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return ""

def test_chatbot_responses(csv_file):
    pass_count = 0
    fail_count = 0

    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    for row in rows:
        user_input = row["Input"]
        expected_output = row["Expected Output"]
        bot_response = get_bot_response(user_input)

        print(f"Testing input: {user_input}")
        print(f"Expected Output: {expected_output}")

        if isinstance(bot_response, list):
            bot_response_str = ' '.join([str(item) for item in bot_response])
        else:
            bot_response_str = bot_response

        print(f"Bot Response: {bot_response_str}")

        result = "PASS" if bot_response_str.strip() == expected_output.strip() else "FAIL"
        print("Test Passed: ", result)
        print("-" * 50)

        row["Result"] = result
        if result == "PASS":
            pass_count += 1
        else:
            fail_count += 1

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ["Concept", "Input", "Expected Output", "Result"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    current_time = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nSummary:")
    print(f"Test Conducted By: {username}")
    print(f"Time of execution: {current_time}")
    print(f"Total PASS cases: {pass_count}")
    print(f"Total FAIL cases: {fail_count}")
    if fail_count == 0:
        print(f"__________________________________")
        print(f"GOOD JOB {username}🥳🥳🥳!!! All test cases are cleared")
    else:
        print(f"__________________________________")
        print(f"Sorry {username}😞😞😞, Please check the code once again. \n There are {fail_count} Cases failed.")

create_csv(csv_file)
test_chatbot_responses(csv_file)