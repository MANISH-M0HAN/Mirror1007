import csv
import os

def request_response(user_input, response):
    file_path = "Debug/request_response_stack.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(["User Input", "Response"])
        
        writer.writerow([user_input, response])
