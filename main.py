import LucasTen as codes

def __main__():
    # Step 1: Load Training and Testing Data
    test_top, test_mid, test_bot, test_label, test_three = codes.load_data('MNISTdata/MNISTval.csv')
    
    # Step 2: Train the Models
    print("Training Models...")
    odd_even_model, digit_model = codes.train_models('MNISTdata/MNISTtrain.csv')

    # Step 3: Predict Using the Models
    print("Predicting with Trained Models...")
    predictions = codes.predict_pipeline(odd_even_model, digit_model, test_top, test_mid, test_bot)

    # Step 4: Evaluate Predictions
    print("Predictions:", predictions)
    print("True Labels:", test_label)
    accuracy = (predictions == test_label).mean()
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

    # Step 5: Identify and Display Incorrect Predictions
    incorrect_indices = [i for i, (pred, true) in enumerate(zip(predictions, test_label)) if pred != true]
    print(f"Number of Incorrect Predictions: {len(incorrect_indices)}")

    # for idx in incorrect_indices:
    #     img = test_three[idx]
    #     true_label = test_label[idx]
    #     predicted_label = predictions[idx]
    #     print(f"True: {true_label}, Predicted: {predicted_label}")
    #     A4codes.plotImg(img)

# Call the main function
if __name__ == "__main__":
    __main__()