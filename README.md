# Skincare Recommendation System

## Overview

This project provides a personalized skincare recommendation system. It uses machine learning to analyze skincare products and recommend the best options based on user-selected skin types.

---

## Features

- Multi-label classification for skincare product suitability.
- Personalized product recommendations.
- Modular and maintainable code structure.
- Uses VotingClassifier for an ensemble learning approach.

---

## Setup and Usage

### 1. **Set Up the Environment**

#### Create a Virtual Environment (Optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Required Packages

Make sure you have `pip` installed, then run:

```bash
pip install -r requirements.txt
```

---

### 2. **Place the Dataset**

Ensure the `cosmetics.csv` file is located in the `datasets/` directory.

---

### 3. **Run the Main Script**

Execute the main script to start the recommendation system:

```bash
python main.py
```

---

### 4. **Interact with the Recommendation System**

- The script will prompt you to select your skin type(s).
- Input the numbers corresponding to your skin types, separated by commas.

#### Example:
```text
Welcome to the Skincare Recommendation System!
Please select your skin type(s) from the following options:
1. Combination
2. Dry
3. Normal
4. Oily
5. Sensitive

Enter the numbers corresponding to your skin types, separated by commas: 1,4
```

---

### 5. **View Recommendations**

After entering your skin types, the system will display the top recommended products for you.

#### Example Output:

```text
Recommended Products:
           Brand                                                               Name    Price  Rank    Score
  DRUNK ELEPHANT                                          T.L.C. Sukari Babyfacial™ 0.209809  0.90 0.873749
FIRST AID BEAUTY                Hello FAB Coconut Skin Smoothie Priming Moisturizer 0.068120  0.84 0.870646
FIRST AID BEAUTY                                      Ultra Repair Face Moisturizer 0.057221  0.84 0.870146
         EVE LOM                                                      Moisture Mask 0.237057  0.76 0.870016
    IT COSMETICS Secret Sauce Clinically Advanced Miraculous Anti-Aging Moisturizer 0.177112  0.76 0.868481
```

---

## Project Structure

```plaintext
your_project/
├── data_processing.py       # Data loading and preprocessing functions
├── model_training.py        # Model definition, training, and saving
├── recommendation.py        # Recommendation logic using the trained model
├── main.py                  # Main script to interact with the user
├── requirements.txt         # List of required Python packages
└── datasets/
    └── cosmetics.csv        # Dataset file
```

---

## Key Notes

- **Virtual Environment**: It is recommended to use a virtual environment to isolate dependencies.
- **Dataset**: Ensure the dataset (`cosmetics.csv`) is in the correct folder (`datasets/`).
- **Requirements**: Install all required packages using `pip install -r requirements.txt`.
- **Error Handling**: The script handles basic errors, such as invalid user input or no suitable products found.
- **Modular Design**: The code is modular, allowing easy modifications and additions.

---

## Future Enhancements

- Improve model performance using advanced techniques like feature engineering or deep learning.
- Add support for more product features or additional datasets.
- Enhance user interface for better interactivity.

---

## License

This project is open-source and free to use under the [MIT License](https://opensource.org/licenses/MIT).