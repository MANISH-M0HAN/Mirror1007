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
    ["Single trigger, Multiple Intent", "what is angina? and why is it caused", "Angina is chest pain caused by reduced blood flow to the heart. It signals an increased risk of heart attack."],
    ["Single Synonym", "what is lipid?", "Cholesterol is a type of fat found in your blood."],
    ["Single Synonym, Multiple Intent", "what is lipid? and why is it caused", "Cholesterol is a type of fat found in your blood. High levels increase the risk of heart disease."],
    ["Single Keyword", "what is heartbeat?", "Tachycardia is a rapid heartbeat."],
    ["Single Keyword, Multiple Intent", "what is heartbeat? and how to prevent", "Tachycardia is a rapid heartbeat. Avoid stimulants, manage stress, and follow medical advice."],
    ["Multiple Synonyms","What is Rapid and Plaque","Atherosclerosis is the buildup of plaque in the arteries. \n\n Tachycardia is a rapid heartbeat."],
    ["Multiple Synonyms, Multiple Intent","How is Rapid Prevented and Why is Rhythm caused?","Avoid caffeine, manage stress, and take medications as prescribed. It can lead to serious complications like stroke. \n\n Avoid stimulants, manage stress, and follow medical advice. It can increase the risk of stroke or heart failure."],
    ["Multiple Keywords","What is Palpitations and infection","Arrhythmia is an irregular heartbeat. \n\n Endocarditis is the inflammation of the heart's inner lining."],
    ["Multiple Keywords, Multiple Intent","How is Arterial prevented and What is heartbeat","Maintain a healthy lifestyle with exercise, a balanced diet, and regular check-ups. Cardiovascular refers to the heart and blood vessels. \n\n Avoid stimulants, manage stress, and follow medical advice. Tachycardia is a rapid heartbeat."],
    ["Combination of Trigger and Synonym with multiple intent","What is Cholesterol and How is Inflammation prevented","Cholesterol is a type of fat found in your blood. Eat a low-fat diet, exercise regularly, and take prescribed medications. \n\n Endocarditis is the inflammation of the heart's inner lining. Prevent infections, maintain dental hygiene, and take antibiotics if needed."],
    ["Combination of Trigger and Synonym without multiple intent","What is Cardiovascular and Narrowing","Cardiovascular refers to the heart and blood vessels. \n\n Stenosis is the narrowing of blood vessels or heart valves."],
    ["Combination of Trigger and Keyword with multiple intent","what is Myocardial and symptoms of Palpitations","Arrhythmia is an irregular heartbeat. Palpitations, dizziness, fainting. \n\n Myocardial infarction is a heart attack. Chest pain, shortness of breath, nausea."],
    ["Combination of Trigger and Keyword without multiple intent","why is obstruction and Tachycardia caused","It can restrict blood flow and lead to heart failure. \n\n It can increase the risk of stroke or heart failure."],
    ["Combination of Keyword and Synonym with multiple intent","How is Coronary prevented and Why is Discomfort caused","Maintain a healthy lifestyle with exercise, a balanced diet, and regular check-ups. Understanding cardiovascular health is key to preventing heart disease in women. \n\n Manage stress, avoid heavy meals, and follow a healthy lifestyle. It signals an increased risk of heart attack."],
    ["Combination of Keyword and Synonym without multiple intent","Symptoms of Heartbeat and Rhythm","Palpitations, dizziness, fainting. \n\n Rapid heartbeat, dizziness, shortness of breath."],
    ["Combination of Trigger, Synonym and Keyword without intent","How is hypertension , Narrowing and infection prevented","Regular monitoring, reducing salt intake, and managing stress. \n\n Regular check-ups and treatment to manage symptoms. \n\n Prevent infections, maintain dental hygiene, and take antibiotics if needed."],
    ["Combination of Trigger, Synonym and Keyword with intent","What is cholesterol How is rapid prevented and symptoms of discomfort","Cholesterol is a type of fat found in your blood. Eat a low-fat diet, exercise regularly, and take prescribed medications. No symptoms, but detected through blood tests. \n\n Tachycardia is a rapid heartbeat. Avoid stimulants, manage stress, and follow medical advice. Rapid heartbeat, dizziness, shortness of breath. \n\n Angina is chest pain caused by reduced blood flow to the heart. Manage stress, avoid heavy meals, and follow a healthy lifestyle. Chest pain, shortness of breath, nausea."],
    ["Only trigger word", "angina", "Angina is chest pain caused by reduced blood flow to the heart.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
    ["Only trigger word - multiple", "angina cardiovascular", "Cardiovascular refers to the heart and blood vessels. \n\n Angina is chest pain caused by reduced blood flow to the heart.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
    ["Only synonym word", "lipid", "Cholesterol is a type of fat found in your blood.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
    ["Only synonym word - multiple", "narrowing plaque", "Atherosclerosis is the buildup of plaque in the arteries. \n\n Stenosis is the narrowing of blood vessels or heart valves.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
    ["Only keyword word", "obstruction", "Stenosis is the narrowing of blood vessels or heart valves.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
    ["Only keyword word - multiple", "hypertensive heartbeat","Hypertension is high blood pressure. \n\n Tachycardia is a rapid heartbeat.\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."],
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
        payload = {"user-input": user_input}
        response = requests.post(url, headers=headers, json=payload)
        return response.json().get("data", {}).get("chatbot-response")
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