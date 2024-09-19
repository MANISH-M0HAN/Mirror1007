import csv
import os

def request_response(user_input,corrected_input, response):
    file_path = "Debug/request_response_stack.csv"
    file_exists = os.path.isfile(file_path)
    
    #Open the CSV file in append mode
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header if thr file does not exist
        if not file_exists:
            writer.writerow(["User Input","Corrected Input", "Response"])
        
        # Append the new row with user input, corrected input, and response
        writer.writerow([user_input, corrected_input, response])
