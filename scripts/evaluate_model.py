'''Evaluate the model using accuracy score and confusion matrix.'''

from food_recommendation_model import FoodRecommendationModel
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

model = FoodRecommendationModel()
model.load_datasets("data\\foods.csv", "data\\users.csv", "data\\interactions.csv")
acc, y_test, y_pred = model.train_model() 

cm = confusion_matrix(y_test, y_pred)

labels = ['Dislike (0)', 'Like (1)']
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
