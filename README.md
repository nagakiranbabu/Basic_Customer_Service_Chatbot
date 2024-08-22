# Basic Customer Service Chatbot

## Overview

This project involves the development of a customer service chatbot using Python. The chatbot is designed to classify user intents and generate appropriate responses based on the input it receives. The model uses a Naive Bayes Classifier for intent classification and is implemented with both a command-line training script and a Tkinter-based GUI for interaction.

## Features

- **Intent Classification:** Uses a Naive Bayes Classifier to predict the intent of user messages.
- **Response Generation:** Provides relevant responses based on the predicted intent.
- **Training Data:** Models are trained using a dataset of intents and patterns.
- **Interactive GUI:** Provides a Tkinter-based interface for real-time interactions with the chatbot.

## Technologies Used

- **Python:** Core programming language for implementing the chatbot and training model.
- **NLTK:** Used for natural language processing tasks such as tokenization and lemmatization.
- **Tkinter:** Used for creating the GUI for chatbot interaction.
- **Pickle:** Used for saving and loading the trained model.
- **JSON:** Used for handling the intents data.

## Project Structure

1. **`chatbot.py`**: Contains the implementation of the Naive Bayes Classifier, intent prediction, and GUI setup.
2. **`train_model.py`**: Handles the training of the Naive Bayes Classifier and saves the trained model along with vocabulary and classes.
3. **`intents.json`**: JSON file containing training data including patterns and responses for different intents.

## Getting Started

### Prerequisites

- Python 3.x
- NLTK library (`pip install nltk`)
- Tkinter (usually included with Python)
- `intents.json` file with training data

### Running the Chatbot

1. **Train the Model:**

   Run the `train_model.py` script to train the Naive Bayes Classifier and save the model.

   ```bash
   python train_model.py
   ```

2. **Start the Chatbot:**

   Run the `chatbot.py` script to launch the GUI and interact with the chatbot.

   ```bash
   python chatbot.py
   ```

3. **Interact with the Chatbot:**

   Use the Tkinter GUI to enter messages and receive responses from the chatbot.

## Example Usage

1. Start the chatbot using the GUI.
2. Enter a message in the text box and click "Send."
3. The chatbot will respond based on the trained model and the intents defined in `intents.json`.

## Contributions

Feel free to fork this repository and contribute improvements or new features. For bug reports or feature requests, please open an issue.

## Project Highlights

1. Developed a customer service chatbot using Naive Bayes Classifier for intent classification and response generation, leveraging a dataset for training.
2. Incorporated Object-Oriented Programming (OOP) and Database Management System (DBMS) principles to efficiently manage chatbot interactions and data.
