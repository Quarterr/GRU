
import scipy.io as sio

# Load the .mat file
mat_data = sio.loadmat('predictions_and_true_values.mat')

# Access the data
predictions = mat_data['predictions']
true_values = mat_data['true_values']

# Now you can work with the predictions and true values as NumPy arrays
print(predictions)
print(true_values)

for i in range(0,2000):
    print(predictions[i]-true_values[i])
   # print("prediction:", predictions[i], "true_value",true_values[i])