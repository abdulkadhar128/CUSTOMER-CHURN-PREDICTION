import pickle

# Predict using the tuned model
y_pred = best_model.predict(X_test_scaled if best_model_name in ["Logistic Regression", "SVM", "KNN"] else X_test)

# Generate Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Tuned {best_model_name} - Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("✅ Model saved successfully as best_model.pkl")
