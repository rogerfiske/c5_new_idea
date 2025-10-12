---
title: "Visualizing XGBoost Models with SHAP in Python: Feature Importance, Dependence, and Interaction…"
source: "https://medium.com/top-python-libraries/visualizing-xgboost-models-with-shap-in-python-feature-importance-dependence-and-interaction-1669ff9969b9"
published: 2025-09-11
clipdate: "2025-10-08T12:13:57-07:00"
"created-date": "2025-10-08T12:13:57-07:00"
description: "This article discusses a method for reproducing journal figures using the XGBoost and SHAP frameworks in Python, including a detailed walkthrough of model building, evaluation, and visualizations. It emphasizes automated reporting and feature importance analysis."
---
>[!summary]- Summary


[Sitemap](https://medium.com/sitemap/sitemap.xml)## [Top Python Libraries](https://medium.com/top-python-libraries?source=post_page---publication_nav-d565f18bf45f-1669ff9969b9---------------------------------------)

In this issue, we’re reproducing a figure from a research paper by building a powerful script that unites XGBoost and SHAP. It doesn’t just automatically train a high-performance model; it generates a complete, stunning, and data-packed visualization report at the click of a button! Everything is automated: from data loading and preprocessing to model training and tuning. We use the XGBoost algorithm to guarantee top-tier predictive performance and the SHAP framework to completely demystify the model’s internal logic. Generate dozens of beautiful charts — including feature importance, dependence plots, and interaction effect plots — all with a single click.

## Processing flow

- **User Configuration:** Specify the data file path and the target variable name.
- **Data Preparation:** The script automatically loads the Excel data and intelligently handles non-numeric features to clear the way for model training.
- **Model Building:** The data is split into training and testing sets, and Grid Search (GridSearchCV) is used to automatically find the optimal hyperparameters for the XGBoost model and complete the training.
- **Performance Evaluation:** Key metrics such as R², RMSE, and MAE are calculated, and an intuitive regression fit plot is generated to clearly display the model’s performance.
- **Core SHAP Analysis:** Calculates the SHAP values and SHAP interaction values for the features, which is the key to unlocking the model’s “black box.”
- **Results Visualization:** Batch-generates a series of in-depth analysis charts, including: the SHAP feature importance summary plot, the SHAP interaction summary plot, SHAP dependence plots for all features, and interaction plots for all pairs of features.
- **Results Export:** Automatically saves all charts and detailed interaction data to a specified folder, making it convenient for you to write reports or perform further analysis.

## Result

***Paper: Uncovering the multiple socio-economic driving factors of carbon emissions in nine urban agglomerations of China based on machine learning***

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*HtKFJ7p15RmmMj5mx6cMuQ.jpeg)

***Imitation***

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*PvpmH36Ol4qox2m0eqfKMQ.png)

## Code implementation (segmented explanation)

1. **User Configuration**
- This is the only place you need to modify! The soul entrance of the script.
- excel\_file\_path: points to the Excel file, the heart of your data. Please use raw string literals (r’’) or replace backslashes (\\) in the path with / to avoid escape character issues.
- target\_column\_name: tells the script explicitly which column is the target (Y value) you care about and want to predict.
- sheet\_name: If your Excel has multiple worksheets, please specify the sheet name here.
```hs
# =============================================================================
# ===== 0. User Configuration (Please modify your file paths and column names here) =====
# =============================================================================
excel_file_path = r'D:\folder\data.xlsx'  # Define the full path to the Excel data file
target_column_name = 'Vegetation_Coverage'  # Define the name of the column in your data that is the target variable (Y value)
sheet_name = None  # If the Excel file has multiple worksheets, please specify the name of the sheet to read here; if there is only one or to read the first one, keep it as None
```

**2\. Preparation**

```hs
# =============================================================================
# ===== 1. Preparation: Import libraries, load and prepare data =====
# =============================================================================
import pandas as pd  # Import pandas library for data processing and analysis, with DataFrame as its core
import numpy as np  # Import numpy library for efficient numerical calculations
import matplotlib.pyplot as plt  # Import matplotlib's pyplot module for plotting charts
import matplotlib  # Import the main matplotlib library for configuration
import os  # Import the os library to interact with the operating system, e.g., creating folders, joining paths
import warnings  # Import the warnings library to control the display of warning messages
import re  # Import the re library for regular expression operations, used here for cleaning filenames
from statsmodels.nonparametric.smoothers_lowess import lowess  # Import the lowess smoothing function from statsmodels for plotting smooth fitted curves
import shap  # Import the shap library for model interpretation, calculating SHAP values
from sklearn.model_selection import train_test_split, GridSearchCV  # Import data splitting and grid search cross-validation tools from sklearn
import xgboost as xgb  # Import the xgboost library, an efficient gradient boosting framework
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Import evaluation metric calculation functions from sklearn
import matplotlib.colors as mcolors
matplotlib.use('TkAgg')  # Set the matplotlib backend, 'TkAgg' is a commonly used graphical interface backend
plt.rcParams['font.serif'] = ['Times New Roman']  # Set the serif font used by matplotlib for plotting to 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']  # Set the sans-serif font used by matplotlib for plotting to 'SimHei' (a Chinese font) to display Chinese characters correctly
plt.rcParams['axes.unicode_minus'] = False  # Set matplotlib to display the minus sign correctly
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore specific types of runtime warnings to avoid unnecessary output
def sanitize_filename(name):  # Define a function to clean illegal characters from filenames
    return re.sub(r'[\\/*?:"<>|]', '_', name)  # Use regular expressions to replace illegal characters in Windows filenames with underscores
print("--- Starting task ---")  # Print a message indicating the start of the task
print(f"Loading data from '{excel_file_path}'...")  # Print a message indicating that the data file is being loaded
if sheet_name:  # Check if the user specified a worksheet name
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)  # If specified, read the specific worksheet
else:  # If no worksheet name is specified
    df = pd.read_excel(excel_file_path)  # Default to reading the first worksheet of the Excel file
print("Data loaded successfully!")  # Print a success message if the data is loaded successfully
if target_column_name not in df.columns:  # Check if the target column exists in the DataFrame's column names
    print(f"Error: Target column '{target_column_name}' does not exist in the data!")  # If not, print an error message
    print(f"Available columns are: {df.columns.tolist()}")  # And list all available column names for the user to check
    exit()  # Exit the program
y = df[target_column_name]  # Assign the data from the target column to the variable y
X = df.drop(columns=[target_column_name])  # Drop the target column from the DataFrame, and assign the rest as the feature set to X
feature_names = X.columns.tolist()  # Get the names of all features and store them in a list
print(f"Target (y) has been set to the '{target_column_name}' column.")  # Print a confirmation that the target column has been set
print(f"Found a total of {len(feature_names)} features for model training: {feature_names}")  # Print the number and names of the features found
print("Checking and processing non-numeric features...")  # Print a message indicating the start of data preprocessing
for col in X.columns:  # Iterate over all feature columns
    if X[col].dtype == 'object':  # Check if the data type of the column is 'object' (usually a string)
        print(f" -> Feature '{col}' is a non-numeric type, attempting to process it.")  # If so, print a message
        X_converted = pd.to_numeric(X[col], errors='coerce')  # Try to convert the column to a numeric type, 'coerce' will turn unconvertible values into NaN
        if X_converted.isnull().all():  # If all values become NaN after conversion, it means the column is purely text
            print(f" -> Feature '{col}' is a purely text categorical feature, will use factorize for encoding.")  # Print a message indicating that factorize encoding will be used
            X[col], _ = pd.factorize(X[col])  # Use pandas' factorize method to convert text to numeric codes
        else:  # If some values can be converted
            X[col] = X_converted  # Assign the converted column (containing NaNs) back to its place
            if X[col].isnull().sum() > 0:  # Check if there are any NaN values resulting from failed conversions
                median_val = X[col].median()  # Calculate the median of the column
                print(f" -> Feature '{col}' contains some unconvertible values, which will be filled with the median ({median_val:.2f}).")  # Print a message indicating that NaNs will be filled with the median
                X[col].fillna(median_val, inplace=True)  # Fill NaN values with the median
print("Data preprocessing complete.")  # Print a message indicating that data preprocessing is complete
```

**3\. Model training and evaluation**

- Data partitioning: The dataset is divided into training and test sets in a ratio of 7:3. random\_state=0 ensures that the results of each partition are the same, making it easy to reproduce the experiment.
- Hyperparameter search: Parameters are crucial to determining the quality of a model. Here, we define a parameter grid, param\_grid, which contains the core parameters we want to try, such as n\_estimators (number of trees), max\_depth (depth of the tree), and learning\_rate (learning rate).
- GridSearchCV: This is an automated parameter tuning tool. It tries all possible parameter combinations and evaluates each combination using 3-fold cross-validation (cv=3), ultimately finding the optimal parameter set with the highest score (here, the lowest negative mean squared error). n\_jobs=-1 means using all CPU cores for parallel computation, significantly speeding up the search.
- Model Evaluation and Plotting: Use the best model found to make predictions on both the training and test sets, and calculate R², RMSE, and MAE metrics. Finally, plot the true and predicted values in a scatter plot to visually demonstrate the model fit and automatically save the results.
```hs
# =============================================================================
# ===== 2. Dataset Splitting, Hyperparameter Search & Model Training (Model is XGBoost) =====
# =============================================================================
print("\nSplitting dataset into training and validation sets...") # Print a message indicating the start of dataset splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # Use train_test_split to split the data, 30% as the test set
print(f"Dataset splitting complete. Training set: {X_train.shape[0]} samples, Validation set: {X_test.shape[0]} samples.") # Print the number of samples in the resulting splits
print("\nPerforming hyperparameter search for the XGBoost model...") # Print a message indicating the start of hyperparameter search
param_grid = { # Define a dictionary containing the hyperparameters and their candidate values to search
    'n_estimators': [100, 200, 500], # Number of trees
    'max_depth': [5, 10, 15], # Maximum depth of the trees
    'learning_rate': [0.05, 0.1, 0.2] # Learning rate
}
xgb_model = xgb.XGBRegressor(random_state=0, eval_metric='rmse') # Initialize an XGBoost regressor model, set random seed and evaluation metric
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=1) # Configure grid search, using 3-fold cross-validation, parallel processing, and printing progress information
grid_search.fit(X_train, y_train) # Execute grid search on the training set
model = grid_search.best_estimator_ # Get the best model found by the grid search
print("\nHyperparameter search complete!") # Print a message indicating the completion of the search
print(f"The best parameters found are: {grid_search.best_params_}") # Print the best hyperparameter combination found
print("The final model has been built using the best parameters.") # Print that the best model is being used
print("\nPlotting the regression fit for the training and validation sets...") # Print a message indicating the start of plotting the fit graph
y_train_pred = model.predict(X_train) # Use the model to predict on the training set
y_test_pred = model.predict(X_test) # Use the model to predict on the test set
# Calculate metrics for the training set
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)
# Calculate metrics for the validation set
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f"\nTraining set evaluation metrics: R2={r2_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")
print(f"Validation set evaluation metrics: R2={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
# ====================================

plt.figure(figsize=(8, 8), dpi=150) # Create a new figure window and set its size and resolution
plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train', color='blue') # Plot a scatter plot of actual vs. predicted values for the training set
plt.scatter(y_test, y_test_pred, alpha=0.7, label='Validation', color='red', marker='^') # Plot a scatter plot of actual vs. predicted values for the test set
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='1:1 Fit Line (y=x)') # Plot a y=x reference line
plt.xlabel('Actual Values', fontsize=12) # Set the X-axis label
plt.ylabel('Predicted Values', fontsize=12) # Set the Y-axis label
plt.title('XGBoost', fontsize=14) # Set the chart title
plt.legend(loc='upper left') # Display the legend in the upper left corner
plt.grid(True) # Display grid lines
# Create a text box to be displayed on the plot
metrics_text = (
    f'Validation Set:\n'
    f'$R^2$ = {r2_test:.4f}\n'
    f'RMSE = {rmse_test:.4f}\n'
    f'MAE = {mae_test:.4f}\n\n'
    f'Training Set:\n'
    f'$R^2$ = {r2_train:.4f}\n'
    f'RMSE = {rmse_train:.4f}\n'
    f'MAE = {mae_train:.4f}'
)
# Place the text box in the bottom right corner of the plot
plt.text(0.95, 0.05, metrics_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
# =======================================
output_main_folder = r'D:\folder' # Define the path for the main output folder
os.makedirs(output_main_folder, exist_ok=True) # Create the main output folder, do not raise an error if it already exists
fit_plot_path = os.path.join(output_main_folder, 'regression_fit_plot.png') # Construct the full save path for the fit plot
plt.savefig(fit_plot_path, bbox_inches='tight') # Save the figure, bbox_inches='tight' crops the white border
print(f"Regression fit plot has been saved to: '{fit_plot_path}'") # Print the save path
plt.show() # Display the figure
```

**4\. SHAP Analysis Preparation**

- Create an explainer: shap.Explainer(model) initializes a SHAP explainer using our best trained model.
- explainer(X\_test) computes the main effect SHAP value for each test sample and each feature.
- explainer.shap\_interaction\_values(X\_test) computes the more expensive but more informative interaction effect SHAP values. This is a 3D array that records the interaction effect between each pair of features.
- Feature sorting: Sort each feature by its global importance based on its mean absolute SHAP value (mean\_shap), and arrange all subsequent data (X\_test\_sorted, shap\_values\_sorted, etc.) according to this new order to ensure that the Y-axis order of subsequent charts is consistent and meaningful.
```hs
# =============================================================================
# ===== 3. SHAP Analysis (Key Correction: All calculations are based on X_test) =====
# =============================================================================
print("\nCalculating SHAP values on the *test set* using the best model...") # Print a message indicating the start of SHAP value calculation on the test set
explainer = shap.Explainer(model) # Create a SHAP explainer using the trained best model
print("Calculating main effect SHAP values (based on X_test)...") # Print a message indicating the start of main effect SHAP value calculation
shap_values_obj = explainer(X_test) # Pass the test set data to the explainer to calculate SHAP values for each feature of each sample
shap_values = shap_values_obj.values # Extract the SHAP value matrix from the SHAP explanation object
print("Main effect SHAP values calculation complete.") # Print a message indicating the calculation is complete
print("\nCalculating SHAP interaction values (based on X_test)...") # Print a message indicating the start of interaction effect SHAP value calculation
shap_interaction_values = explainer.shap_interaction_values(X_test) # Calculate SHAP interaction values, which is a 3D array
print("SHAP interaction values calculation complete.") # Print a message indicating the calculation is complete
print("\nSorting features based on SHAP values...") # Print a message indicating the start of feature sorting
mean_shap = np.abs(shap_values).mean(axis=0) # Calculate the mean of the absolute SHAP values for each feature as its importance measure
shap_df = pd.DataFrame({ # Create a DataFrame to store feature names and their importance
    "feature": feature_names, # Column for feature names
    "mean_shap": mean_shap # Column for mean absolute SHAP values
}).sort_values("mean_shap", ascending=False) # Sort in descending order of importance
sorted_features = shap_df["feature"].values # Get the list of feature names after sorting
print("Feature sorting complete.") # Print a message indicating the sorting is complete
X_test_sorted = X_test[sorted_features] # Reorder the columns of the test set according to the sorted feature order
orig_index = [feature_names.index(f) for f in sorted_features] # Get the indices of the sorted features in the original feature list
shap_values_sorted = shap_values[:, orig_index] # Reorder the columns of the SHAP value matrix according to the new feature order
shap_interaction_values_sorted = shap_interaction_values[:, orig_index][:, :, orig_index] # Similarly, reorder the rows and columns of the interaction value matrix
```

**5\. General auxiliary functions**

- This section defines three reusable “Swiss Army Knife” functions that are the cornerstones of subsequent advanced drawing.
- bootstrap\_lowess\_ci: This ensures statistical robustness. It uses the bootstrap resampling technique to calculate a 95% confidence interval for the LOWESS smoothed fit curve. This ensures that the trend line we see is no longer an isolated estimate, but rather a statistically reliable range of fluctuations.
- find\_and\_plot\_crossings: Automatically finds the intersections (thresholds) of the curve with the y=0 line and marks them on the plot with dashed lines and labels. This helps us quickly locate the key points where the feature influence changes from positive to negative (or vice versa).
- find\_roots: A plot-less version of find\_and\_plot\_crossings that simply computes and returns the values of the intersection points.
```hs
# =============================================================================
# ===== General Helper Functions =====
# =============================================================================

def bootstrap_lowess_ci(x, y, n_boot=200, frac=0.5, ci_level=0.95):
    """
    Calculates the confidence interval for a LOWESS smooth using the bootstrap method.

    Args:
        x (pd.Series): The input feature for the model (independent variable).
        y (pd.Series): The model's output or true value (dependent variable).
        n_boot (int): The number of bootstrap samples to generate. A higher number results in a more
                      stable confidence interval estimate but increases computational cost. Defaults to 200.
        frac (float): The fraction of the data used in the LOWESS smoother. This value controls the
                      degree of smoothing and should be between 0 and 1. A smaller value results in a
                      curve that fits the data points more closely; a larger value results in a
                      smoother curve. Defaults to 0.5.
        ci_level (float): The confidence level for the interval. For example, 0.95 corresponds to a
                          95% confidence interval. Defaults to 0.95.

    Returns:
        tuple: A tuple containing (main_smoothed_curve, (x_range, lower_bound, upper_bound)),
               or (None, None) if the calculation cannot be performed.
    """
    if len(x) < 10: return None, None # If there are too few data points, do not perform the calculation.
    
    boot_lines = [] # Initialize a list to store the smoothed curve from each bootstrap sample.
    x_range = np.linspace(x.min(), x.max(), 100) # Generate 100 evenly spaced points across the range of x for interpolation.
    
    for _ in range(n_boot): # Loop for the specified number of bootstrap samples.
        # Draw sample indices with replacement.
        sample_indices = np.random.choice(len(x), len(x), replace=True)
        # Get the sample data using the indices.
        x_sample, y_sample = x.iloc[sample_indices], y.iloc[sample_indices]
        
        # Sort the sampled x-values.
        sorted_indices = np.argsort(x_sample)
        # Get the sorted x and y values.
        x_sorted, y_sorted = x_sample.iloc[sorted_indices].values, y_sample.iloc[sorted_indices].values
        
        if len(np.unique(x_sorted)) < 2: continue # If there are fewer than 2 unique x-values in the sample, skip this iteration.
        
        # Perform LOWESS smoothing on the sample data.
        smoothed = lowess(y_sorted, x_sorted, frac=frac)
        # Interpolate the smoothed results onto the common x_range.
        interp_func = np.interp(x_range, smoothed[:, 0], smoothed[:, 1])
        # Add the interpolated curve to the list.
        boot_lines.append(interp_func)
        
    if not boot_lines: return None, None # If no bootstrap curves were generated, return None.
    
    # Sort the original x data.
    sorted_indices_orig = np.argsort(x)
    # Get the sorted original x and y values.
    x_sorted_orig, y_sorted_orig = x.iloc[sorted_indices_orig].values, y.iloc[sorted_indices_orig].values
    
    # Perform LOWESS smoothing on the complete original data to get the main curve.
    main_smoothed = lowess(y_sorted_orig, x_sorted_orig, frac=frac)
    
    # Convert the list of bootstrap curves to a NumPy array.
    boot_lines_arr = np.array(boot_lines)
    
    # Calculate alpha for the confidence interval.
    alpha = (1 - ci_level) / 2
    # Calculate the lower and upper confidence bounds at each point.
    lower_bound = np.quantile(boot_lines_arr, alpha, axis=0)
    upper_bound = np.quantile(boot_lines_arr, 1 - alpha, axis=0)
    
    # Return the main smoothed curve and the confidence interval data.
    return main_smoothed, (x_range, lower_bound, upper_bound)
```

**6\. SHAP overview and dependency graph drawing**

- SHAP Feature Importance Overview Chart: Here, through the clever twiny() (shared Y-axis) technique, the bar chart representing global importance and the swarm plot representing local details are perfectly superimposed together to create a composite chart with extremely high information density.
- SHAP interaction summary plot: Call shap.summary\_plot directly, but pass in the interaction effect value. It will show the feature pairs with the strongest interaction effect.
- Batch Dependency Plots (plot\_shap\_dependence): Defines a powerful plotting function that generates a five-in-one dependency plot for a single feature, including scatter plots, distribution histograms, fitted curves, confidence intervals, and threshold markers. A for loop then automatically calls this function for all features, generating and saving these plots in batches.
```hs
# =============================================================================
# ===== 4. Plotting SHAP Summary and Dependence Plots =====
# =============================================================================
print("\nPlotting SHAP feature importance summary plot...") # Print a message indicating the start of summary plot creation.
fig = plt.figure(figsize=(10, 10), dpi=300) # Create a new figure window.
ax_sw = fig.add_axes([0.32, 0.11, 0.59, 0.77]) # Add an axes object to the figure for the swarm plot.
ax_bar = ax_sw.twiny() # Create a second x-axis that shares the same y-axis, to be used for the bar plot.
ax_bar.set_zorder(0); # Place the bar plot axes in the background.
ax_sw.set_zorder(1); # Place the swarm plot axes in the foreground.
ax_sw.patch.set_alpha(0) # Make the background of the top axes transparent.

y_pos = np.arange(len(sorted_features))[::-1] # Calculate the y-axis position for each feature.
# Plot a horizontal bar chart showing the mean importance of each feature.
ax_bar.barh(y=y_pos, width=shap_df["mean_shap"].values, height=0.6, color="blue", alpha=0.5, edgecolor="none", zorder=0)
xlim_bar = shap_df["mean_shap"].values.max() * 1.05 # Calculate the upper limit for the bar plot's x-axis.
ax_bar.set_xlim(0, xlim_bar) # Set the x-axis range for the bar plot.
xticks_bar = np.linspace(0, xlim_bar, 5) # Calculate the tick locations for the bar plot's x-axis.
ax_bar.set_xticks(xticks_bar) # Set the x-axis ticks for the bar plot.
ax_bar.set_xticklabels([f"{x:.2f}" for x in xticks_bar]) # Set the format for the bar plot's x-axis tick labels.
ax_bar.set_xlabel("Mean (|SHAP| value)", fontsize=10) # Set the label for the bar plot's x-axis.
ax_bar.set_yticks(y_pos) # Set the y-axis tick positions (shared with the swarm plot).

max_abs_shap = np.abs(shap_values_sorted).max() # Calculate the maximum absolute SHAP value.
xlim_sw = max_abs_shap * 1.1 # Calculate the range for the swarm plot's x-axis.
ax_sw.set_xlim(-xlim_sw, xlim_sw) # Set the x-axis range for the swarm plot.
sw_xticks = np.linspace(-xlim_sw, xlim_sw, 5) # Calculate the tick locations for the swarm plot's x-axis.
ax_sw.set_xticks(sw_xticks) # Set the x-axis ticks for the swarm plot.
ax_sw.set_xticklabels([f"{x:.2f}" for x in sw_xticks]) # Set the format for the swarm plot's x-axis tick labels.
ax_sw.set_xlabel("SHAP value (impact on model output)", fontsize=10) # Set the label for the swarm plot's x-axis.

# Create a SHAP Explanation object, which is the standard format for the new shap library's plotting functions.
expl_main = shap.Explanation(
    values=shap_values_sorted, # The sorted SHAP values.
    data=X_test_sorted.values, # The sorted feature values.
    feature_names=list(sorted_features), # The sorted feature names.
    base_values=shap_values_obj.base_values[0] # The model's base value.
)
# Draw the beeswarm plot on the specified axes.
shap.plots.beeswarm(expl_main, max_display=len(sorted_features), ax=ax_sw, show=False, plot_size=None)

ax_sw.set_yticks(y_pos) # Set the y-axis tick positions.
ax_sw.set_yticklabels(sorted_features, fontsize=12) # Set the y-axis tick labels (feature names).
ax_sw.tick_params(axis='y', length=4) # Adjust the length of the y-axis tick marks.

# ===== Remove the top and right spines from the plot frame =====
ax_sw.spines['top'].set_visible(False)
ax_sw.spines['right'].set_visible(False)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
# ===============================================================

combined_image_path = os.path.join(output_main_folder, 'combined_shap_summary_plot.png') # Construct the save path for the combined plot.
plt.savefig(combined_image_path, dpi=208, bbox_inches='tight') # Save the figure.
print(f"SHAP feature importance summary plot saved to: '{combined_image_path}'") # Print a confirmation message.
plt.show() # Display the plot.

print("\nPlotting SHAP interaction summary plot...") # Print a message indicating the start of interaction plot creation.
plt.figure() # Create a new figure.
# Use the shap library to plot a summary of interaction values.
shap.summary_plot(shap_interaction_values_sorted, X_test_sorted, max_display=10, show=False)
interaction_summary_plot_path = os.path.join(output_main_folder, 'shap_interaction_summary_plot.png') # Construct the save path for the interaction plot.
plt.savefig(interaction_summary_plot_path, dpi=300, bbox_inches='tight') # Save the figure.
print(f"SHAP interaction summary plot saved to: '{interaction_summary_plot_path}'") # Print a confirmation message.

def plot_shap_dependence(feature_name, x_values, shap_values_for_feature, save_folder, custom_annotation=None):
    """
    Plots and saves a SHAP dependence plot for a single feature.

    **SHAP Value Scatter Plot**: Shows the relationship between each sample's feature value and its
                                corresponding SHAP value (blue scatter points).
    **Feature Value Distribution Histogram**: Displays the distribution of the feature in the dataset
                                           as a background bar chart (purple bars).
    **LOWESS Smoothed Fit Curve**: Reveals the average trend of how the SHAP value changes with the
                                 feature value (dark violet solid line).
    **Confidence Interval**: Provides a range of statistical reliability for the LOWESS curve,
                           typically a 95% confidence interval (light gray filled area).
    **Threshold (Intersection) Labeling**: Automatically finds and marks the points where the fitted
                                         curve crosses y=0. These points are key thresholds where the
                                         feature's influence changes direction (positive/negative)
                                         (black dashed lines and labels).
    **Custom Annotation**: Allows the user to add custom text to the plot.

    Args:
        feature_name (str): The name of the feature to be plotted. Used for the x-axis label and
                            as part of the output filename.
        x_values (pd.Series or np.array): All sample values for this feature in the dataset.
        shap_values_for_feature (np.array): The SHAP values corresponding one-to-one with x_values.
        save_folder (str): The folder path where the generated image file will be saved.
        custom_annotation (dict, optional): An optional dictionary to add a custom annotation to the plot.
                                            Example: {'text': 'Key Region', 'x': 0.8, 'y': 0.8}
    """
    print(f"  -> Plotting feature: {feature_name} ...") # Print which feature is currently being plotted.
    fig_dep, ax1 = plt.subplots(figsize=(8, 6), dpi=150) # Create a new figure and axes.
    ax2 = ax1.twinx() # Create a second y-axis that shares the x-axis.
    ax2.patch.set_alpha(0) # Make the background of the second y-axis transparent.

    # Calculate histogram data for the feature's distribution.
    counts, bin_edges = np.histogram(x_values, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # Calculate the center of each bin.
    bin_width = bin_edges[1] - bin_edges[0] # Calculate the width of each bin.
    # Plot the distribution histogram on ax1.
    ax1.bar(bin_centers, counts, width=bin_width * 0.6, align='center', color='#4B0082', alpha=0.3, label='Distribution')
    ax1.set_ylabel('Distribution', fontsize=12) # Set the y-axis label for ax1.
    ax1.set_ylim(0, counts.max() * 1.1) # Set the y-axis range for ax1.

    # Plot the scatter plot of SHAP values on ax2.
    ax2.scatter(x_values, shap_values_for_feature, alpha=0.3, s=25, color='#00008B', label='Sample', zorder=2)
    
    if len(x_values) > 1: # Check if there are enough samples.
        # Calculate the LOWESS smoothed curve and its confidence interval.
        main_fit, ci_data = bootstrap_lowess_ci(x_values, shap_values_for_feature, frac=0.3)
        if main_fit is not None and ci_data is not None: # If the calculation was successful.
            ax2.plot(main_fit[:, 0], main_fit[:, 1], color='#9400D3', lw=2, label='LOWESS Fit', zorder=4) # Plot the main smoothed curve.
            ax2.fill_between(ci_data[0], ci_data[1], ci_data[2], color='#D3D3D3', alpha=0.15, label='95% CI') # Fill the confidence interval.
            ax2.axhline(0, color='black', linestyle='--', lw=1, zorder=1) # Draw a reference line at y=0.
            find_and_plot_crossings(ax2, main_fit[:, 0], main_fit[:, 1], 'black') # Find and plot the threshold lines.
            
    ax2.set_ylabel('SHAP value', fontsize=12) # Set the y-axis label for ax2.
    y_max = np.abs(shap_values_for_feature).max() * 1.15 # Calculate the y-axis range for ax2.
    if y_max < 1e-6: y_max = 1 # Avoid a range that is too small.
    ax2.set_ylim(-y_max, y_max) # Set the y-axis range for ax2.
    
    ax1.set_xlabel(f'{feature_name}', fontsize=12) # Set the shared x-axis label.
    
    if custom_annotation and isinstance(custom_annotation, dict): # Check for a custom annotation.
        text = custom_annotation.get('text', ''); # Get the annotation text.
        x_pos = custom_annotation.get('x', 0.95); # Get the annotation x-coordinate.
        y_pos = custom_annotation.get('y', 0.95) # Get the annotation y-coordinate.
        # Define the style of the annotation.
        props = {'ha': custom_annotation.get('ha', 'right'), 'va': custom_annotation.get('va', 'top'),
                 'fontsize': custom_annotation.get('fontsize', 12), 'color': custom_annotation.get('color', 'darkred'),
                 'fontweight': custom_annotation.get('fontweight', 'bold')}
        ax1.text(x_pos, y_pos, text, transform=ax1.transAxes, **props) # Add the custom annotation to the plot.
        
    h1, l1 = ax1.get_legend_handles_labels(); # Get legend handles and labels from ax1.
    h2, l2 = ax2.get_legend_handles_labels() # Get legend handles and labels from ax2.
    ax2.legend(h2 + h1, l2 + l1, loc='upper right', fontsize=10) # Combine and display the legends from both y-axes.

    sanitized_feature_name = sanitize_filename(feature_name) # Sanitize the feature name for use in a filename.
    output_filename = f"dependence_plot_{sanitized_feature_name}.png" # Construct the output filename.
    full_path = os.path.join(save_folder, output_filename) # Construct the full save path.
    fig_dep.savefig(full_path, dpi=200, bbox_inches='tight') # Save the figure.
    plt.close(fig_dep) # Close the figure to free up memory.

print("\nStarting batch generation of dependence plots for all features...") # Print a message indicating the start of batch plotting.
output_folder_dep = os.path.join(output_main_folder, 'dependence_plots') # Define the output folder path for dependence plots.
os.makedirs(output_folder_dep, exist_ok=True) # Create the folder.
print(f"All dependence plots will be saved to the '{output_folder_dep}' folder.") # Print a confirmation of the save location.

for i, feature_name in enumerate(sorted_features): # Loop through all sorted features.
    x_data_loop = X_test_sorted[feature_name] # Get the data for the current feature.
    if not np.isfinite(x_data_loop).all(): # Check if the feature contains non-finite values like NaN or infinity.
        print(f"  -> Skipping feature: '{feature_name}' because it contains non-finite values (e.g., NaN).") # If so, skip this feature.
        continue # Continue to the next loop iteration.
        
    y_data_shap_loop = shap_values_sorted[:, i] # Get the SHAP values corresponding to the current feature.
    annotation_for_this_plot = None # Initialize annotation as None.
    
    # if i == 0: # Example: If this is the first (most important) feature
    #     # Add a special annotation for it.
    #     annotation_for_this_plot = {'text': 'This is the most important feature!', 'x': 0.98, 'y': 0.98,
    #                                 'color': 'purple'}
    # elif i == 1: # Example: If this is the second feature
    #     # Add another different annotation.
    #     annotation_for_this_plot = {'text': 'Second most important', 'x': 0.05, 'y': 0.2, 'ha': 'left',
    #                                 'color': 'green'}
    
    # Call the plotting function to create and save the dependence plot.
    plot_shap_dependence(feature_name=feature_name, x_values=x_data_loop, shap_values_for_feature=y_data_shap_loop,
                         save_folder=output_folder_dep, custom_annotation=annotation_for_this_plot)

print(f"\nTask complete! All dependence plots have been successfully generated and saved.") # Print a final completion message.
```
![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*itEgaX586uht7Wm5LZKKeQ.png)

**SHAP summary plot of model features**

- This figure shows a summary of the SHAP (SHapley Additive exPlanations) values of all input features in the model, aiming to reveal the global importance of each feature and the distribution of its impact on the model output.
- Each row in the figure represents a feature, sorted from top to bottom by its global importance (average SHAP absolute value). Therefore, “slope” is the most important feature in this model, while “wind speed” is relatively less important. The bottom x-axis shows the SHAP value of the feature in each example, which measures the magnitude and direction of the feature’s marginal contribution to a single prediction; positive values indicate a positive impact on the model output, while negative values indicate a negative inhibitory effect.
- Each dot in the figure represents a sample, and its color indicates the original numerical value of the feature in the corresponding sample. The color spectrum gradually changes from blue (low value) to red (high value), as shown in the color bar on the right. The horizontal position distribution of the points reveals the relationship between feature value and model influence. For example, for the “slope” and “annual precipitation” features, samples with higher original values (red points) generally have positive SHAP values, while samples with lower original values (blue points) correspond to negative SHAP values, showing a positive correlation trend. The top X-axis (Mean |SHAP| value) clearly indicates the average absolute SHAP value of each feature, which is used to quantify its global importance.
![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*QlkTCGKqxwwQLlokxgZ3ig.png)

**SHAP dependency graph of the “light intensity” feature**

- This plot reveals how the marginal contribution (SHAP value) of the feature “light intensity” changes as its own value changes.
- The X-axis in the figure represents the raw value of “light intensity”. The Y-axis on the right is the SHAP value of the corresponding sample, which shows the impact of “light intensity” on the model output.
- Each lavender dot represents an independent sample. The solid purple line is a trend line fitted using the LOWESS (Locally Weighted Scatter Point Smoothing) method, reflecting the average trend of the SHAP value; the surrounding light gray shaded area is the 95% confidence interval (CI). The vertical gray bar in the background corresponds to the left Y-axis (Distribution), showing the distribution of “light intensity” samples in the dataset.
- Analysis shows a significant nonlinear relationship between “light intensity” and model output. When light intensity falls below approximately -2.37, it has a positive impact on model predictions. Within the range of -2.37 to -0.11, the impact turns negative. As light intensity increases from -0.11, the impact turns positive again, peaking near 1.0 before weakening. The vertical dashed lines in the figure mark the critical thresholds (-2.37 and -0.11) where the impact shifts, providing precise quantitative insights into the mechanism of action of this feature.

**7\. SHAP interaction value export**

Charts provide intuitive insights, but the raw data is equally important. This section saves the detailed SHAP values of each primary feature interacting with all other features as separate CSV files. This provides a valuable foundation for deeper quantitative analysis or custom plots.

```hs
# =============================================================================
# ===== 5. Extract and save detailed SHAP interaction values for each feature =====
# =============================================================================
print("\nExtracting detailed, per-sample SHAP interaction values for each feature...") # Print a message indicating the start of interaction value extraction
output_folder_interactions = os.path.join(output_main_folder, 'interaction_values_per_feature') # Define the output folder for interaction value CSV files
os.makedirs(output_folder_interactions, exist_ok=True) # Create the folder
print(f"Detailed interaction values for each feature will be saved to the '{output_folder_interactions}' folder.") # Print a message with the save path
for i, primary_feature_name in enumerate(sorted_features): # Iterate through each feature, treating it as the "primary feature"
    interaction_data = {} # Initialize a dictionary to store the interaction values of this primary feature with all other features
    for j, secondary_feature_name in enumerate(sorted_features): # Iterate through all features again, as the "secondary feature"
        if i == j: continue # Skip if the primary and secondary features are the same
        column_name = f"interaction_{sanitize_filename(primary_feature_name)}_vs_{sanitize_filename(secondary_feature_name)}" # Construct the column name
        interaction_values_for_pair = shap_interaction_values_sorted[:, i, j] # Extract the interaction values for this pair of features from the interaction effect matrix
        interaction_data[column_name] = interaction_values_for_pair # Store the interaction values in the dictionary
    feature_interaction_df = pd.DataFrame(interaction_data) # Convert the dictionary to a DataFrame
    print(f"  -> Processing primary feature: '{primary_feature_name}' and saving its interaction values...") # Print which primary feature is being processed
    csv_filename = f"interactions_for_{sanitize_filename(primary_feature_name)}.csv" # Construct the CSV filename
    full_path = os.path.join(output_folder_interactions, csv_filename) # Concatenate the full save path
    feature_interaction_df.to_csv(full_path, index=False, encoding='utf-8-sig') # Save the DataFrame as a CSV file, using 'utf-8-sig' encoding for compatibility with Excel (for Chinese characters)
print(f"\nTask complete! Detailed interaction values for all {len(sorted_features)} features have been saved as separate CSV files.") # Print a task completion message
if len(sorted_features) > 0: # Check if there are any features
    first_feature_name = sorted_features[0] # Get the name of the most important feature
    print(f"\nPreview: Interaction values for the most important feature '{first_feature_name}' with other features (first 5 rows):") # Print a preview message
    first_feature_csv_path = os.path.join(output_folder_interactions,
                                          f"interactions_for_{sanitize_filename(first_feature_name)}.csv") # Get the corresponding CSV file path
    if os.path.exists(first_feature_csv_path): # Check if the file exists
        preview_df = pd.read_csv(first_feature_csv_path) # Read the CSV file
        print(preview_df.head()) # Print the first 5 rows of the file as a preview
```

**8\. SHAP interactive diagram drawing**

This is the grand finale of the script and also its most insightful section.

- plot\_advanced\_interaction function: This is the most complex plotting function in the entire script, which combines all the previous techniques into one.
- Scatter plot: The main feature is the X-axis, the interactive SHAP value is the Y-axis, and the color is determined by the interactive feature.
- Grouped Fitting: Based on the median of the interaction feature, the samples are divided into a “high value group” and a “low value group,” and LOWESS fitting curves with confidence intervals are drawn for each group. This clearly reveals the interaction effect.
- Common Threshold: Automatically finds and marks the stable threshold point where both sets of curves cross y=0.
- Layout fine-tuning: Manually create the color bar axes through fig.add\_axes and precisely control their position and size. This solves problems such as legend compression and shortened color bars that may occur during matplotlib’s automatic layout, ensuring the professionalism and aesthetics of the final output.
- A double loop: the outer loop iterates over each main feature, while the inner loop iterates over each interaction feature, calling the plot\_advanced\_interaction function to generate advanced interaction plots for all possible feature combinations in the data (N\*(N-1)). This will be a computationally intensive but highly rewarding process!
```hs
# =============================================================================
# ===== 6. Plot advanced SHAP interaction graphs =====
# =============================================================================
print("\n--- Starting advanced interaction plot generation task (final revised version) ---") # Print a message indicating the start of the advanced plotting task

# Define a function to plot advanced interaction graphs between two features
def plot_advanced_interaction(primary_feature_name, interacting_feature_name, x_values, interaction_feature_values, shap_interaction_slice, save_folder):
    """
    Plots and saves an advanced, information-rich feature interaction SHAP plot.
    This function aims to visualize how the SHAP value of a primary feature is influenced by an interacting feature.
    The plot mainly consists of the following parts:

    1.  **Interaction Scatter Plot**: The primary feature's value is on the X-axis, and the SHAP interaction value is on the Y-axis. The color of the scatter points
        is determined by the value of the interacting feature, using the 'seismic' (blue-white-red) colormap to intuitively show how high or low values of the
        interacting feature affect the primary feature's impact.
    2.  **Grouped Fit Curves**: The interacting feature's data is split into "high value" and "low value" groups based on its median. LOWESS smooth fit curves
        (red and blue solid lines) and their confidence intervals are then plotted for each group. This clearly reveals whether the trend of the primary feature's
        effect changes based on the level of the interacting feature.
    3.  **Common Threshold Calibration**: Automatically calculates and finds a "stable" threshold point where both group's fit curves cross y=0. If found, it is
        marked on the plot with a purple dashed line and a label. This threshold may represent a robust point of effect transition that is not influenced by the
        interacting feature.
    4.  **Background Distribution Plot**: A gray bar chart in the background shows the data distribution of the primary feature, providing a data density
        reference for trend analysis.

    Parameters:
    primary_feature_name (str): Name of the primary feature, to be displayed on the X-axis.
    interacting_feature_name (str): Name of the interacting feature, whose values will determine scatter point color and grouping.
    x_values (pd.Series or np.array): Array of the actual values for the primary feature.
    interaction_feature_values (pd.Series or np.array): Array of the actual values for the interacting feature.
    shap_interaction_slice (np.array): Array of the corresponding SHAP interaction values between the primary and interacting features.
    save_folder (str): Folder path where the generated image files will be saved.
    """
    sanitized_primary, sanitized_interacting = sanitize_filename(primary_feature_name), sanitize_filename(interacting_feature_name) # Sanitize the names of the primary and interacting features
    print(f"  -> Plotting: '{primary_feature_name}' (Interacting with: '{interacting_feature_name}')") # Print which pair of features is being plotted
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150) # Create a new figure and subplot
    ax2 = ax1.twinx() # Create a second y-axis that shares the x-axis

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "#4B0082", "red"])
    points = ax2.scatter(x_values, shap_interaction_slice, c=interaction_feature_values,
                         cmap=cmap, alpha=1, s=25, zorder=2, label='sample') # Plot the scatter plot on ax2, with point color determined by the interacting feature's value
    
    median_val = interaction_feature_values.median() # Calculate the median of the interacting feature values
    low_mask, high_mask = interaction_feature_values <= median_val, interaction_feature_values > median_val # Create two boolean masks based on the median for grouping
    
    # --- Modification 1: Correct the legend label text ---
    groups = { # Define configuration for the two groups
        'low': {'mask': low_mask, 'color': 'blue', 'offset': 0.9,
                'label': f' {interacting_feature_name} <= {median_val:.2f}'}, # Configuration for the 'low' value group, including a dynamically generated legend label
        'high': {'mask': high_mask, 'color': 'red', 'offset': 0.8,
                 'label': f' {interacting_feature_name} > {median_val:.2f}'} # Configuration for the 'high' value group
    }
    
    counts, bin_edges = np.histogram(x_values, bins=30) # Calculate the distribution data for the primary feature
    bin_centers, bin_width = (bin_edges[:-1] + bin_edges[1:]) / 2, bin_edges[1] - bin_edges[0] # Calculate the bin centers and width
    ax1.bar(bin_centers, counts, width=bin_width * 0.7, color='gray', alpha=0.2, label='Distribution') # Plot the distribution histogram on ax1
    ax1.set_ylabel('Distribution', fontsize=12) # Set the y-axis label for ax1
    ax1.set_ylim(0, counts.max() * 1.1) # Set the y-axis range for ax1
    
    fits, roots = {}, {} # Initialize dictionaries to store fit curves and roots
    for name, info in groups.items(): # Iterate through the 'low' and 'high' value groups
        x_group, shap_group = x_values[info['mask']], shap_interaction_slice[info['mask']] # Get the group data using the mask
        if len(x_group) < 10: continue # If the group has too few samples, skip it
        
        main_fit, ci_data = bootstrap_lowess_ci(x_group, shap_group) # Perform LOWESS smoothing and confidence interval calculation for this group's data
        if main_fit is not None and ci_data is not None: # If the calculation is successful
            ax2.plot(main_fit[:, 0], main_fit[:, 1], color=f'dark{info["color"]}', lw=2.5,
                     label=info['label'])  # <-- Plot the smooth curve using the corrected label
            ax2.fill_between(ci_data[0], ci_data[1], ci_data[2], color=info['color'], alpha=0.15) # Fill the confidence interval for this curve
            fits[name] = main_fit # Store the fitted curve
            roots[name] = find_roots(main_fit[:, 0], main_fit[:, 1]) # Calculate and store the roots of this curve

    if 'low' in roots and 'high' in roots: # If roots were found for both the 'low' and 'high' groups
        tolerance = (x_values.max() - x_values.min()) * 0.05 # Define a tolerance to determine if two roots are "close"
        for r_low in roots['low']: # Iterate through the roots of the 'low' group
            for r_high in roots['high']: # Iterate through the roots of the 'high' group
                if abs(r_low - r_high) < tolerance: # If the two roots are very close
                    avg_root = (r_low + r_high) / 2 # Calculate their average
                    ax2.axvline(x=avg_root, color='black', linestyle='--', linewidth=1) # Draw a purple vertical dashed line at this position to indicate a common threshold
                    ax2.text(avg_root, ax2.get_ylim()[1] * 0.9, f' {avg_root:.2f} ', color='white',
                             backgroundcolor='purple', ha='center', va='center', fontsize=9,
                             bbox=dict(facecolor='purple', edgecolor='none', pad=1)) # Add a text label above the line

    # --- Adjust the spacing of the Color Bar ---
    # Manually create a new Axes at a specific position on the Figure, dedicated to the color bar.
    # This method provides the most precise layout control.
    # The coordinates [left, bottom, width, height] are ratios relative to the entire figure (from 0 to 1).
    #   left=0.975: The left edge of the color bar starts at 97.5% from the figure's left side, placing it to the right of the main plot with some gap.
    #   bottom=0.11, height=0.77: Defines the vertical position and length of the color bar, aligning it vertically with the main plot's axes, which solves the issue of the color bar becoming shorter automatically.
    #   width=0.02: Defines the width of the color bar; this value now solely controls the "thickness" of the color bar.
    cbar_ax = fig.add_axes([0.975, 0.11, 0.02, 0.77])
    # Plot the colorbar into the dedicated axes \`cbar_ax\` we just created.
    #   points: This is the scatter plot object (returned by ax.scatter), and the color bar will be drawn based on its colormap (cmap) and data range.
    #   cax=cbar_ax: This \`cax\` parameter is key; it tells matplotlib to "draw the color bar in this specified cbar_ax," instead of letting it automatically find a position.
    cbar = fig.colorbar(points, cax=cbar_ax)
    
    cbar.set_label(f"Value of {interacting_feature_name}", size=12) # Set the label for the color bar
    ax1.set_xlabel(f'{primary_feature_name}', fontsize=12) # Set the x-axis label
    ax2.set_ylabel(f'SHAP Interaction Value', fontsize=12) # Set the y-axis label for ax2
    # fig.suptitle(f"Interaction: {primary_feature_name} vs {interacting_feature_name}", fontsize=14) # Set the main title for the entire figure
    ax2.axhline(0, color='black', linestyle='--', lw=1, zorder=0) # Draw the y=0 reference line
    
    y_max_abs = np.abs(shap_interaction_slice).max() * 1.1 # Calculate the y-axis range for ax2
    ax2.set_ylim(-y_max_abs if y_max_abs > 1e-6 else -1, y_max_abs if y_max_abs > 1e-6 else 1) # Set the y-axis range for ax2
    ax2.legend(loc='best', fontsize=10) # Display the legend
    
    ax1.set_zorder(0); # Set the z-order of ax1 to the bottom layer
    ax2.set_zorder(1); # Set the z-order of ax2 to the top layer
    ax2.patch.set_alpha(0) # Set the background of ax2 to transparent
    
    output_filename = f"interaction_{sanitized_primary}_vs_{sanitized_interacting}.png" # Construct the output filename
    full_path = os.path.join(save_folder, output_filename) # Concatenate the full save path
    fig.savefig(full_path, dpi=200, bbox_inches='tight') # Save the figure
    plt.close(fig) # Close the figure to release memory

output_folder_advanced_interactions = os.path.join(output_main_folder, 'advanced_interaction_plots_final') # Define the output folder for the advanced interaction plots
os.makedirs(output_folder_advanced_interactions, exist_ok=True) # Create the folder if it doesn't exist
print(f"\nAll advanced interaction plots will be saved to the '{output_folder_advanced_interactions}' folder.") # Print the save path message

n_features = len(sorted_features) # Get the total number of features
for i in range(n_features): # Iterate through all features as the primary feature
    primary_feature_name = sorted_features[i] # Get the primary feature name
    for j in range(n_features): # Iterate through all features again as the interacting feature
        if i == j: continue # If it's the same feature, skip it
        
        interacting_feature_name = sorted_features[j] # Get the interacting feature name
        x_values = X_test_sorted[primary_feature_name] # Get the values of the primary feature
        interaction_feature_values = X_test_sorted[interacting_feature_name] # Get the values of the interacting feature (for coloring)
        shap_interaction_slice = shap_interaction_values_sorted[:, i, j] * 2 # Get the interaction SHAP values for this pair (multiplied by 2 for consistency with shap library's default behavior)
        
        plot_advanced_interaction( # Call the plotting function for the advanced interaction graph
            primary_feature_name=primary_feature_name,
            interacting_feature_name=interacting_feature_name,
            x_values=x_values,
            interaction_feature_values=interaction_feature_values,
            shap_interaction_slice=shap_interaction_slice,
            save_folder=output_folder_advanced_interactions
        )
print(f"\n--- All plotting tasks are complete! ---") # Print the final completion message
```
![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*fv6FzXg1jtXo4u4_g69DOA.png)

**SHAP interaction effect diagram of “yearly rainfall” and “wind speed”**

- This figure details how the impact of “yearly rainfall” on the model prediction is moderated by the interactive feature “wind speed”.
- The X-axis in the figure shows the value of the primary feature, “annual precipitation,” and the Y-axis on the right shows the SHAP interaction effect between “yearly rainfall” and “wind speed.” The color of the scattered points in the figure is determined by the value of the interactive feature, “wind speed,” as shown in the color bar on the right: red represents high wind speed and blue represents low wind speed.
- To clearly analyze the interaction effect, all samples were divided into a high wind speed group (red trend line) and a low wind speed group (blue trend line) based on the median wind speed (0.11). Both trend lines are LOWESS fitted curves, and the shaded areas around them are their respective 95% confidence intervals. Observe that the two trend lines show diametrically opposite trends:
1. **Under high wind speed conditions (red curve)**, the SHAP interaction effect value is roughly positively correlated with annual precipitation, indicating that when annual precipitation increases, its combined effect with high wind speed has a stronger positive contribution to the model output.
2. **Under low wind speed conditions (blue curve)**, the SHAP interaction effect value is negatively correlated with annual precipitation, indicating that when annual precipitation increases, its combined effect with low wind speed has a negative impact on the model output.

This significant divergence reveals a strong interaction between annual precipitation and wind speed. Furthermore, the gray bars in the background show the distribution of annual precipitation, while the vertical dashed line at -0.02 marks a potential threshold for the interaction effect to shift.

Thank you for reading.

[![Top Python Libraries](https://miro.medium.com/v2/resize:fill:96:96/1*d3JXV6YjxMmjIYTctmmdqQ.png)](https://medium.com/top-python-libraries?source=post_page---post_publication_info--1669ff9969b9---------------------------------------)

[![Top Python Libraries](https://miro.medium.com/v2/resize:fill:128:128/1*d3JXV6YjxMmjIYTctmmdqQ.png)](https://medium.com/top-python-libraries?source=post_page---post_publication_info--1669ff9969b9---------------------------------------)

[Last published 13 hours ago](https://medium.com/top-python-libraries/11-python-libraries-that-make-cybersecurity-feel-effortless-85b416f39322?source=post_page---post_publication_info--1669ff9969b9---------------------------------------)

## More from ZHEMING XU and Top Python Libraries

## Recommended from Medium

[

See more recommendations

](https://medium.com/?source=post_page---read_next_recirc--1669ff9969b9---------------------------------------)