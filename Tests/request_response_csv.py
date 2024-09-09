import csv
import os

<<<<<<< HEAD
def request_response(user_input,corrected_input, response):
=======
def request_response(user_input, response):
>>>>>>> c80ff75 (PR for adding env-example and revision of Test_me.py along with minor improvements (#22))
    file_path = "Debug/request_response_stack.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
<<<<<<< HEAD
            writer.writerow(["User Input","Corrected Input", "Response"])
        
        writer.writerow([user_input, corrected_input, response])
=======
            writer.writerow(["User Input", "Response"])
        
        writer.writerow([user_input, response])
>>>>>>> c80ff75 (PR for adding env-example and revision of Test_me.py along with minor improvements (#22))
