import matplotlib.pyplot as plt

def generate_report():
    user_inputs = []
    bot_responses = []

    with open('interactions.log', 'r') as log_file:
        lines = log_file.readlines()
        for i in range(0, len(lines), 3):
            user_inputs.append(lines[i].strip().split(": ")[1])
            bot_responses.append(lines[i+1].strip().split(": ")[1])

    # Example: Generate a simple bar chart of the number of interactions
    plt.bar(range(len(user_inputs)), [len(ui) for ui in user_inputs], tick_label=user_inputs)
    plt.xlabel('User Inputs')
    plt.ylabel('Length of User Inputs')
    plt.title('User Interaction Lengths')
    plt.xticks(rotation=90)
    plt.show()

if __name__ == "__main__":
    generate_report()
