All are working mostly fine based on the tests done in the Terminal. But once they are tested in the POSTMAN client it shows how the context history is messing up the results. 
First thing to do tomorrow is to come up with the new logic for chatbot both With and Without CSV file working.


Current Code working described with ChatGPT

Here's a step-by-step breakdown of the current code's logic:

Steps in Current Code's Logic:
1.User Query Input:
User sends a query through a POST request to the /chatbot endpoint.

2.Spelling Correction:
The correct_spelling function is called to correct any spelling mistakes in the user input.

3.Check for Greetings:
The input is checked to see if it matches common greetings (e.g., "hello", "hi", "hey"). If it does, a standard greeting response is returned.

4.Finding the Best Context:
The find_best_context function is called to match the user query against precomputed embeddings of the CSV data.

5.CSV Data Embeddings:
The CSV file is read and each row is converted into embeddings:
-Trigger word embedding.
-Synonyms embeddings.
-Keywords embeddings.
These embeddings are precomputed and stored in a list called db_embeddings.

6.Cosine Similarity Calculation:
The user query's embedding is computed.
Cosine similarity scores are calculated between the user query embedding and each of the trigger, synonyms, and keywords embeddings in db_embeddings.
The highest similarity score is used to find the best matching response from the database, based on a threshold.

7.Appending Context:
If a context match is found, it is appended to the context_history list.
The user's query is also appended to the context_history.

8.Generating Full Context:
A full context string is created by concatenating all entries in the context_history along with the user's current query.

9.Response Generation:
The generate_response_with_gemini function is called with the full context string to generate a response using the Gemini model.

10.Update Context History:
The generated response is appended to the context_history.
If the context history exceeds 10 entries, it is truncated to keep only the last 10.

11.Returning the Response:
The response is returned to the user.
The context history is stored in the session for future queries.

**Session and Context Handling:
-Session Initialization:
When the user first interacts, the context_history is initialized in the session if it doesn't already exist.
-Session Updates:
After generating a response, the context_history in the session is updated with the new context.

**CSV to Embeddings Conversion:
-Reading CSV:
The CSV file is read into a Pandas DataFrame.
-Processing Rows:
Each row is processed to extract:
	Trigger word.
	Synonyms.
	Keywords.
	Response text.
-Computing Embeddings:
Each trigger word, synonym, and keyword is converted into an embedding using the SentenceTransformer model.
These embeddings are stored in a structured format in the db_embeddings list for efficient lookup during query matching.

**Key Concepts:
1.Spelling Correction					: Ensures that the user query is correctly interpreted by correcting any spelling mistakes.
2.Retrieval-Augmented Generation (RAG)	: Combines the retrieval of relevant information from a precomputed database (based on embeddings and cosine similarity) with generative model responses.
3.Generative Model						: Uses the Gemini model to generate a coherent and contextually relevant response based on the user's query and context history.
4.Session Management					: Maintains a history of user interactions to provide contextually relevant responses in future queries.

By following these steps, the system attempts to match the user's query to the best possible response based on the provided data and context history.
