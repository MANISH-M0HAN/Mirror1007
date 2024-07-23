import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import numpy as np

def generate_report():
    user_inputs = []
    bot_responses = []
    timestamps = []

    with open('interactions.log', 'r') as log_file:
        lines = log_file.readlines()

        # Check if the total number of lines is a multiple of 4
        if len(lines) % 4 != 0:
            print("Warning: The number of lines in the log file isn't a multiple of 4. This may indicate formatting issues.")

        for i in range(0, len(lines), 4):
            try:
                # Handle potential index errors
                timestamp_line = lines[i].strip()
                user_input_line = lines[i + 1].strip()
                bot_response_line = lines[i + 2].strip()

                # Safely split and append to lists
                if ": " in user_input_line and ": " in bot_response_line:
                    timestamp = timestamp_line.split(": ", 1)[1]
                    user_inputs.append(user_input_line.split(": ", 1)[1])
                    bot_responses.append(bot_response_line.split(": ", 1)[1])
                    timestamps.append(timestamp)
                else:
                    print(f"Skipping malformed line pair at index {i}: {user_input_line}, {bot_response_line}")
            except IndexError:
                print(f"Skipping lines due to index error at {i}")
                continue

    # Convert the data into a DataFrame for easier manipulation
    df = pd.DataFrame({'Timestamp': timestamps, 'User Input': user_inputs, 'Bot Response': bot_responses})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    # Plot 1: Length of User Inputs
    plt.figure(figsize=(12, 6))
    df['Input Length'] = df['User Input'].apply(len)
    df['Input Length'].plot(kind='bar', color='skyblue', alpha=0.7)
    plt.xlabel('Interaction Index')
    plt.ylabel('Length of User Inputs')
    plt.title('Length of User Inputs Over Time')
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.show()

    # Plot 2: Common User Queries
    all_words = ' '.join(user_inputs).lower().split()
    common_words = Counter(all_words)
    common_words = dict(common_words.most_common(10))

    plt.figure(figsize=(10, 5))
    plt.bar(common_words.keys(), common_words.values(), color='teal')
    plt.xlabel('Common Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Common Words in User Inputs')
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.show()

    # Plot 3: Word Cloud of User Inputs
    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(user_inputs))
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of User Inputs')
    plt.show()

    # Plot 4: Sentiment Analysis
    df['Sentiment'] = df['User Input'].apply(lambda text: TextBlob(text).sentiment.polarity)
    plt.figure(figsize=(12, 6))
    df['Sentiment'].plot(kind='hist', bins=20, color='coral', alpha=0.7)
    plt.xlabel('Sentiment Polarity')
    plt.title('Sentiment Analysis of User Inputs')
    plt.axvline(df['Sentiment'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.tight_layout()
    plt.show()

    # Plot 5: Interaction Frequency Over Time
    plt.figure(figsize=(12, 6))
    df.resample('H').size().plot(kind='line', color='green', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Number of Interactions')
    plt.title('User Interactions Over Time')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_report()