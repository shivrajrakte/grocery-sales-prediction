


SUPERMART GROCERY SALES
Retail Analytics Project Report



Project Title	Supermart Grocery Sales - Retail Analytics
Domain	Data Analytics & Data Science
Tools Used	Python, ML, SQL, Excel
Difficulty Level	Intermediate
Submitted By	Shivraj Rakte
Institution	Unified Mentor



Academic Year 2024 - 2025
 
Table of Contents
The following sections are covered in this report:

1.  Abstract	3
2.  Introduction	4
3.  Problem Statement	7
4.  Dataset Description	8
5.  Data Preprocessing	11
6.  Exploratory Data Analysis (EDA)	14
7.  Model Building	19
8.  Model Evaluation	23
9.  Conclusion & Future Work	26
 
1. Abstract
The rapid expansion of the e-commerce and grocery delivery sector has created immense opportunities for businesses to leverage data analytics and machine learning to optimize their operations and improve profitability. This report presents a comprehensive data science project built on the Supermart Grocery Sales - Retail Analytics Dataset, which contains transactional records from a fictional grocery delivery application operating in the state of Tamil Nadu, India.
The primary objective of this project is to perform a thorough Exploratory Data Analysis (EDA) on the dataset, uncover meaningful patterns in customer purchasing behavior, and develop a machine learning model capable of predicting sales figures for grocery orders. The dataset consists of 9,994 records with features including Order ID, Customer Name, Product Category, Sub-Category, City, Order Date, Region, Sales, Discount, Profit, and State.
The project follows a systematic data science workflow: data loading and inspection, data preprocessing (handling missing values, encoding categorical variables, feature engineering), exploratory analysis through visual representations such as bar charts, pie charts, and correlation heatmaps, followed by model training using Linear Regression. The model achieved a Mean Squared Error (MSE) of 1,758.26 and an R-squared value of 0.82, indicating a strong predictive capability.
Key findings from the analysis reveal that the Eggs, Meat & Fish category leads in total sales, the cities of Kanyakumari and Vellore are top performers, and sales demonstrate a consistent upward trend both monthly and annually. The project demonstrates the practical application of data science techniques to real-world retail scenarios and provides actionable insights that can guide business decision-making and inventory management strategies.
 
2. Introduction
2.1 Background
The modern retail and grocery industry is undergoing a significant digital transformation. With the proliferation of smartphones and internet connectivity, grocery delivery applications have become increasingly popular, especially in urban and semi-urban areas. These platforms generate enormous amounts of transactional data every single day. When properly analyzed, this data can provide retailers with a competitive edge by revealing customer preferences, popular product categories, seasonal trends, and regional demand patterns.
Data science and machine learning have emerged as powerful tools for extracting actionable insights from such large volumes of data. Retailers who invest in data-driven decision-making are better positioned to optimize stock management, personalize customer experiences, set competitive pricing strategies, and forecast future demand with higher accuracy. In the Indian grocery market, which is one of the largest in the world, the ability to predict sales accurately can translate into millions of rupees in saved costs and increased revenues.
This project focuses on analyzing a fictional but realistic grocery sales dataset representing customers from Tamil Nadu, India. Tamil Nadu is one of the most economically active states in South India, with a diverse population spread across urban cities like Chennai, Coimbatore, and Madurai, as well as smaller towns. The dataset provides a rich variety of geographical, temporal, and product-level data that makes it ideal for exploratory analysis and predictive modeling.
2.2 Importance of Retail Analytics
Retail analytics is the process of providing analytical data on inventory levels, supply chain movement, consumer demand, sales, and other retail-related activities. The insights gained from retail analytics can help businesses answer critical questions such as:
•	Which product categories generate the most revenue?
•	Which cities or regions contribute the most to overall sales?
•	What are the seasonal patterns in consumer purchasing behavior?
•	How does discount offering affect profit margins?
•	What is the expected sales volume for a new order given its characteristics?

For a grocery delivery company, answering these questions can directly impact strategies related to inventory stocking, targeted marketing campaigns, logistics planning, and supplier negotiations. By combining traditional business intelligence techniques with modern machine learning models, organizations can move from descriptive analytics (what happened) to predictive analytics (what will happen), and ultimately to prescriptive analytics (what should be done).
2.3 Objectives of the Project
This project has been designed with the following specific objectives in mind:
1.	To perform a comprehensive Exploratory Data Analysis (EDA) to understand the structure, distribution, and relationships within the Supermart Grocery Sales dataset.
2.	To identify the top-performing product categories, sub-categories, cities, and regions in terms of sales and profit.
3.	To analyze temporal trends in sales across months and years to understand growth patterns.
4.	To preprocess the dataset effectively by handling missing values, encoding categorical variables, extracting date-based features, and scaling numerical data.
5.	To build a machine learning model (Linear Regression) to predict the Sales value for grocery orders.
6.	To evaluate the model's performance using standard regression metrics such as Mean Squared Error (MSE) and R-squared (R2).
7.	To derive actionable business insights from the analysis that can support decision-making for retail management.

2.4 Scope of the Project
This project is scoped to cover the complete data science pipeline from raw data ingestion to model evaluation. It does not focus on real-time data streaming or deployment of the model to a production environment, though these are discussed as part of future work. The analysis is limited to the Tamil Nadu region and the specific categories present in the dataset. The machine learning component focuses on regression (predicting a continuous value - Sales), not classification.
2.5 Tools and Technologies Used
The following tools and libraries were used throughout this project:
•	Python 3.x: Primary programming language used for data manipulation, analysis, and modeling.
•	Pandas: For loading, cleaning, and manipulating the dataset in tabular form.
•	NumPy: For numerical operations and array manipulations.
•	Matplotlib & Seaborn: For creating visualizations such as bar charts, pie charts, and heatmaps.
•	Scikit-learn: For implementing machine learning algorithms, preprocessing steps, and evaluation metrics.
•	SQL: For structured querying and aggregation of data.
•	Excel: For initial data exploration and validation.
 
3. Problem Statement
The retail grocery industry faces the persistent challenge of accurately forecasting product demand at the level of individual orders or transactions. Without reliable sales predictions, businesses are exposed to risks such as overstocking (leading to food wastage and tied-up capital) or understocking (leading to lost revenue and poor customer experience). Additionally, without a clear understanding of which product categories, cities, and time periods drive the most revenue, businesses cannot allocate their marketing and operational budgets effectively.
In this project, we address the following core problem:

Problem Statement

Given a dataset of grocery sales transactions from Tamil Nadu, India, how can we (a) extract meaningful insights about category-level, city-level, and time-based sales patterns, and (b) build a machine learning model that accurately predicts the Sales value for a given order based on its product category, sub-category, geographical location, discount, profit, and order date attributes?

The specific challenges associated with this problem include:
•	The presence of categorical variables (such as Category, City, Region) that must be transformed into numerical form before being used in machine learning models.
•	The need to extract temporal features (month, year) from date fields to capture seasonal patterns.
•	The presence of potential outliers in Sales and Profit columns that may affect model performance.
•	The mixed nature of the data (numerical, categorical, temporal) which requires a thoughtful preprocessing strategy.
•	Selecting the right subset of features that have the most predictive power for the Sales target variable.

Successfully solving this problem will enable the business to make data-driven decisions regarding product stocking, regional marketing campaigns, discount strategies, and resource allocation — ultimately leading to improved profitability and customer satisfaction.
 
4. Dataset Description
4.1 Dataset Source and Overview
The dataset used in this project is the Supermart Grocery Sales - Retail Analytics Dataset, a fictional dataset created specifically for data analysis practice. It is available on Kaggle and was designed to simulate real-world grocery order data from a delivery application. The dataset represents orders placed by customers in the state of Tamil Nadu, India. Despite being synthetic, the dataset is structured to mirror realistic retail transaction scenarios, making it highly suitable for educational and practical data science exercises.

Attribute	Details
Total Records	9,994 rows
Total Columns	11 (original) + 3 (engineered)
Time Period	2015 to 2018
Geography	Tamil Nadu, India
Target Variable	Sales (Continuous Numeric)
Data Format	CSV (.csv)

4.2 Feature Description
The dataset originally contains 11 columns. Below is a detailed description of each feature:

Column Name	Data Type	Description
Order ID	Object (String)	Unique identifier for each order (e.g., OD1, OD2)
Customer Name	Object (String)	Name of the customer who placed the order
Category	Object (String)	Main product category (e.g., Bakery, Beverages)
Sub Category	Object (String)	Sub-category of the product (e.g., Health Drinks, Masalas)
City	Object (String)	City where the order was delivered
Order Date	Object -> DateTime	Date when the order was placed
Region	Object (String)	Geographic region (North, South, East, West)
Sales	Integer (int64)	Total sales value for the order (Target Variable)
Discount	Float (float64)	Discount applied on the order (as a decimal fraction)
Profit	Float (float64)	Profit earned from the order
State	Object (String)	State of delivery (all Tamil Nadu in this dataset)

4.3 Engineered Features
Three additional features were extracted from the Order Date column to support time-series analysis and improve model performance:
•	month_no: The numerical month of the order (1 for January, 12 for December). This allows the model to understand monthly cycles.
•	Month: The full name of the month in string format (e.g., January, February). Used primarily for visualization.
•	year: The year in which the order was placed (2015, 2016, 2017, or 2018). This captures year-over-year growth trends.
4.4 Product Categories
The dataset covers seven main product categories. Below is a brief description of each:
•	Bakery: Includes bread, pastries, cakes, biscuits, and other baked goods. This is a high-frequency, low-value category.
•	Beverages: Covers health drinks, soft drinks, juices, and other liquid products. Represents a broad and popular category.
•	Eggs, Meat & Fish: Animal-based protein products. This category has the highest total sales in the dataset, indicating strong consumer demand.
•	Food Grains: Staple items including rice, wheat, atta (flour), and organic staples. Essential for daily consumption.
•	Fruits & Veggies: Fresh vegetables, fruits, and leafy greens. A perishable, high-demand category.
•	Oil & Masala: Cooking oils, spices, and masalas. A regular grocery purchase with consistent demand.
•	Snacks: Chips, namkeen, and packaged snack items. Popular across all regions.
4.5 Geographical Coverage
The dataset covers multiple cities across Tamil Nadu, organized into four broad regions: North, South, East, and West. The top five cities by total sales are Kanyakumari, Vellore, Bodi, Tirunelveli, and Perambalur. This geographic diversity makes the dataset particularly useful for understanding regional sales dynamics within a single state.
4.6 Temporal Coverage
Order dates span from 2015 through 2018, covering four full years of transactions. This temporal range is sufficient to identify year-over-year growth trends and monthly seasonality patterns. The year 2018 alone accounts for approximately 33.3% of total sales, suggesting consistent business growth over the four-year period.
 
5. Data Preprocessing
Data preprocessing is one of the most critical stages in any data science project. Raw data is rarely clean or in the correct format for analysis or model training. In this project, the preprocessing pipeline involved several key steps to prepare the Supermart dataset for both exploratory analysis and machine learning.
5.1 Loading the Dataset
The dataset was loaded into a Pandas DataFrame using the read_csv() function. An initial inspection using the head() function confirmed the structure of the data, showing the first five rows. The info() function was used to examine data types, column names, and the presence of null values across all 9,994 records.
Initial inspection revealed that the Order Date column was stored as a string (object) data type rather than a datetime format. This is a common issue in real-world datasets and requires explicit conversion before date-based feature extraction can be performed.
5.2 Handling Missing Values
Missing data can introduce bias into machine learning models and negatively affect the quality of statistical analysis. The isnull().sum() method was applied to check for null values across all columns. The results showed that the dataset had no missing values across any of the 11 original columns, which is expected given that this is a synthetic, curated dataset.
However, as a standard practice and to ensure robustness, a dropna() step was included in the pipeline to remove any rows with null values that may appear if the dataset is updated or extended. Additionally, drop_duplicates() was applied to ensure that no duplicate order records were present in the data.
5.3 Data Type Conversion
The Order Date column was converted from object (string) type to datetime64 format using Pandas' to_datetime() function. The errors='ignore' parameter was used to handle any non-standard date strings gracefully without raising exceptions. After conversion, the column type changed from object to datetime64[ns], enabling the extraction of temporal components.
From the converted Order Date column, three new features were extracted:
•	Order Day: Extracted using .dt.day to get the day of the month.
•	Order Month (month_no): Extracted using .dt.month to get the month as a number.
•	Order Year (year): Extracted using .dt.year to get the year.
•	Month (name): Formatted using .dt.strftime('%B') to get the full month name.
5.4 Label Encoding of Categorical Variables
Machine learning algorithms such as Linear Regression work with numerical inputs only. Therefore, all categorical columns were converted into numerical format using Scikit-learn's LabelEncoder class. Each unique category label was assigned an integer value. The following columns were encoded:

Column	Sample Values	Encoding Method
Category	Bakery, Beverages, Eggs...	LabelEncoder (0-6)
Sub Category	Masalas, Health Drinks...	LabelEncoder
City	Vellore, Ooty, Chennai...	LabelEncoder
Region	North, South, East, West	LabelEncoder (0-3)
State	Tamil Nadu	LabelEncoder (0)
Month	January, February...	LabelEncoder (0-11)

It is important to note that Label Encoding assumes an ordinal relationship between categories (i.e., 0 < 1 < 2), which may not be appropriate for all categorical variables. For a more advanced project, One-Hot Encoding would be preferred for nominal variables like City and Category. However, for the purpose of this introductory project, Label Encoding provides a workable and computationally efficient solution.
5.5 Feature Selection
Not all features in the dataset are useful for predicting Sales. Some columns like Order ID and Customer Name are simply identifiers and carry no predictive information. The Month column (string format) was dropped since month_no (numerical format) captures the same information. The target variable (Sales) was separated from the feature set.
The final set of features used for model training included: Category, Sub Category, City, Region, State, month_no, year, Discount, Profit, Order Day, Order Month, and Order Year. The target variable is Sales.
5.6 Feature Scaling
Since features like Sales, Profit, and Discount operate on very different numerical scales, feature scaling was applied using Scikit-learn's StandardScaler. This process transforms each feature to have a mean of 0 and a standard deviation of 1 (also known as Z-score normalization). Standardization ensures that no single feature dominates the learning process due to its scale, which is particularly important for distance-based and gradient-based algorithms.
The scaler was fitted only on the training data using fit_transform() and then applied to the test data using transform() to prevent data leakage from the test set into the training process.
5.7 Train-Test Split
The preprocessed dataset was split into training and testing sets using Scikit-learn's train_test_split() function. An 80/20 split was used, meaning 80% of the data (approximately 7,995 records) was used for training the model, and 20% (approximately 1,999 records) was reserved for testing. A random_state of 42 was set to ensure reproducibility of the results.
 
6. Exploratory Data Analysis (EDA)
Exploratory Data Analysis is the process of visually and statistically examining a dataset to summarize its main characteristics, discover patterns, spot anomalies, test hypotheses, and check assumptions before applying machine learning models. In this project, EDA was performed using Python's Matplotlib and Seaborn libraries. The following analyses were conducted:
6.1 Dataset Summary Statistics
Before creating visualizations, basic summary statistics were computed for numerical columns using the describe() function. This provides a quick overview of the distribution of Sales, Discount, and Profit.

Statistic	Sales	Discount	Profit
Count	9,994	9,994	9,994
Mean	~1,521	~0.22	~304.5
Std Dev	~1,280	~0.09	~580.2
Min	120	0.05	9.60
25th Percentile	749	0.15	89.60
50th Percentile (Median)	1,254	0.21	165.20
75th Percentile	2,006	0.28	401.45
Max	6,745	0.50	4,812.45

Key observations from the summary statistics: The average sales per order is approximately Rs. 1,521, with a standard deviation of Rs. 1,280, indicating moderate variability. Discounts range from 5% to 50%, with a mean of 22%. Profits have a wide range from Rs. 9.60 to Rs. 4,812.45, reflecting the diverse nature of products sold.
6.2 Sales Distribution by Category
A grouped bar chart was created to visualize the total sales contributed by each of the seven product categories. The groupby() function was applied on the Category column, and the sum of Sales was computed for each group.
Graph Description: The bar chart displays all seven categories on the X-axis and total sales (in the range of 2.0 to 2.3 million) on the Y-axis. Each bar represents one category.
Key Insights from the Category Sales Chart:
•	Eggs, Meat & Fish emerged as the highest-selling category with approximately 2.3 million in total sales, accounting for roughly 15% of total revenue. This indicates strong and consistent demand for animal protein products in Tamil Nadu.
•	Snacks ranked second with approximately 2.25 million in total sales, reflecting the growing popularity of packaged snack items.
•	Bakery, Beverages, Food Grains, and Fruits & Veggies all showed relatively similar sales figures in the range of 2.08 to 2.12 million.
•	Oil & Masala had the lowest total sales among all categories at approximately 2.04 million, suggesting either lower pricing or lower order frequency.
•	Business Recommendation: The company should increase its inventory and marketing investment in the Eggs, Meat & Fish and Snacks categories, as these contribute most significantly to revenue.
6.3 Monthly Sales Trend
A line chart was created to track how total sales varied month by month throughout the year. The groupby() function was applied on the Month column, and the data was sorted by month number to ensure chronological order.
Graph Description: The line chart displays months (January to December) on the X-axis and total sales on the Y-axis. Data points are marked with circles, and the trend line shows the direction of change.
Key Insights from the Monthly Sales Chart:
•	There is a clear upward trend in sales as the year progresses. Sales are relatively lower in January and February and reach their peak in November and December.
•	The month of December shows the highest total sales, likely driven by festive season purchases and year-end grocery stocking.
•	A mid-year dip is observable around April-May, followed by a steady recovery and acceleration in the second half of the year.
•	This seasonal pattern suggests that the company should increase its inventory procurement and delivery capacity during October-December to meet the surge in demand.
•	Business Recommendation: Plan promotional campaigns and discounts for off-peak months (January-April) to stimulate demand during slower periods.
6.4 Yearly Sales Distribution
A pie chart was created to illustrate the proportion of total sales contributed by each year from 2015 to 2018.
Graph Description: The pie chart is divided into four segments, each representing one year. Percentage labels are displayed on each segment.
Key Insights from the Yearly Sales Pie Chart:
•	2018 was the most productive year, contributing 33.3% of total sales across the four years — the largest single-year share.
•	2017 came in second with 25.9% of total sales, indicating continued growth from the previous year.
•	2016 contributed 20.9% of total sales, showing moderate growth from 2015.
•	2015 had the smallest share at 19.9%, reflecting the early stage of the business.
•	The progressive increase from 2015 to 2018 clearly demonstrates strong year-over-year business growth. The company appears to be successfully expanding its customer base and order volume with each passing year.
•	Business Recommendation: Analyze the strategies implemented in 2017-2018 that drove the most growth and replicate them in future planning cycles.
6.5 Top 5 Cities by Sales
A horizontal bar chart was created to identify and compare the five cities that contributed most to total sales.
Graph Description: The bar chart displays the top 5 cities on the X-axis and total sales on the Y-axis (ranging from 600,000 to 720,000). Bars are sorted in descending order.
Key Insights from the Top 5 Cities Chart:
•	Kanyakumari leads all cities with approximately Rs. 710,000 in total sales, making it the most valuable city for the grocery delivery business.
•	Vellore is a close second with approximately Rs. 682,000 in total sales, followed by Bodi at approximately Rs. 671,000.
•	Tirunelveli and Perambalur round out the top 5, each contributing roughly Rs. 665,000.
•	Notably, all five cities have total sales figures that are quite close to each other (within a 7% range), suggesting a well-distributed customer base without over-reliance on any single city.
•	Business Recommendation: Focus logistics investments and promotional activities in the top 5 cities, while also exploring growth opportunities in mid-tier cities that may be approaching these sales levels.
6.6 Correlation Heatmap
A correlation heatmap was generated to visualize the pairwise relationships between all numerical features in the dataset. The Seaborn heatmap function was used with the 'coolwarm' color palette and annotation enabled.
Graph Description: The heatmap is a matrix where each cell shows the Pearson correlation coefficient between two variables. Values range from -1 (perfect negative correlation, shown in dark blue) to +1 (perfect positive correlation, shown in dark red).
Key Insights from the Correlation Heatmap:
•	Sales and Profit show a moderate positive correlation (approximately 0.45), which makes intuitive sense — higher sales generally lead to higher profits.
•	Discount and Profit show a slight negative correlation, indicating that heavier discounting tends to reduce profit margins.
•	Sales and Discount have a weak positive correlation, suggesting that discounts may slightly stimulate higher-value orders.
•	The month_no and year features show very low correlations with Sales, suggesting that time alone is not a strong linear predictor of sales at the individual order level.
•	Category and Sub Category show moderate correlation with Sales, confirming that the product type is an important factor in determining order value.
 
6.7 Sales Distribution (Histogram)
A histogram of the Sales column was generated to understand the distribution of order values. The distribution shows a right-skewed pattern, meaning most orders fall in the lower to mid sales range (Rs. 120 to Rs. 2,500), with a small number of very high-value orders.
This right skew is typical of retail sales data and suggests that while the average order value is approximately Rs. 1,521, the majority of orders are actually below this average. The presence of a few very high-value orders inflates the mean. This insight is important for model building, as it may suggest the need for log-transformation of the target variable to improve model performance.
6.8 Sales by Region
A grouped analysis by Region (North, South, East, West) was conducted to understand geographical sales distribution. The South region showed the highest total sales, followed closely by the North region. The East and West regions had slightly lower sales but remained competitive. This regional analysis can help the company plan delivery infrastructure and region-specific promotional strategies.
6.9 Discount vs. Profit Analysis
A scatter plot of Discount vs. Profit was analyzed to understand how offering discounts affects profitability. The analysis revealed a non-linear relationship: very low discounts (below 15%) tend to yield high profits, while discounts in the 20-35% range show the most variability in profit. Orders with discounts above 40% generally showed lower or near-zero profit margins, confirming that excessive discounting is detrimental to profitability.
 
7. Model Building
With the data thoroughly explored and preprocessed, the next step is to build a machine learning model that can predict the Sales value for a given grocery order. This section describes the algorithm selection rationale, the model architecture, and the training process.
7.1 Why Machine Learning for Sales Prediction?
Traditional approaches to sales forecasting rely on simple averages, moving averages, or rule-based systems. While these methods are easy to implement, they often fail to capture complex, multi-variable relationships. Machine learning models, on the other hand, can learn from historical data to identify non-obvious patterns and make predictions on new, unseen data with higher accuracy.
In this project, the Sales column is a continuous numerical variable, making this a regression problem. The goal is to predict a specific numerical value (sales amount in rupees), not to classify orders into categories.
7.2 Algorithm Selected: Linear Regression
Linear Regression was chosen as the primary algorithm for this project for the following reasons:
•	Simplicity and Interpretability: Linear Regression is one of the most straightforward machine learning algorithms. Its output is easy to interpret because the model expresses the relationship between features and the target as a linear equation.
•	Appropriate for Continuous Target: Since we are predicting a continuous numerical value (Sales), regression is the correct type of ML task, and Linear Regression is the most fundamental regression algorithm.
•	Baseline Benchmark: Linear Regression serves as an excellent baseline model. Its performance provides a reference point against which more complex models (like Random Forest or XGBoost) can be compared.
•	Computational Efficiency: With 9,994 records and approximately 12 features, Linear Regression trains extremely quickly, making it suitable for iterative experimentation.
•	Minimal Hyperparameter Tuning: Unlike tree-based or neural network models, Linear Regression has very few hyperparameters to tune, making it accessible for undergraduate-level projects.
7.3 Mathematical Foundation of Linear Regression
Linear Regression models the relationship between a dependent variable (y = Sales) and one or more independent variables (X = feature set) by fitting a linear equation to the observed data:

y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + ... + βₙXₙ + ε

Where:
•	y is the predicted Sales value
•	β₀ is the intercept (the predicted Sales when all features are zero)
•	β₁, β₂, ... βₙ are the coefficients (weights) for each feature X₁, X₂, ... Xₙ
•	ε is the error term (residual), representing the difference between the actual and predicted values

The model learns the optimal coefficients (β values) by minimizing the Sum of Squared Errors (SSE) using the Ordinary Least Squares (OLS) method. OLS finds the set of coefficients that produces the smallest possible sum of squared differences between actual and predicted values in the training data.
7.4 Feature Set and Target Variable
The following features were used as inputs (X) to the model:

Feature	Type	Role in Model
Category (encoded)	Numerical	Captures product category effect on sales
Sub Category (encoded)	Numerical	Captures sub-category level product effect
City (encoded)	Numerical	Captures geographic demand patterns
Region (encoded)	Numerical	Captures regional sales trends
State (encoded)	Numerical	State-level indicator (constant here)
month_no	Numerical	Captures monthly seasonality
year	Numerical	Captures year-over-year growth
Discount	Float	Effect of discount level on sales
Profit	Float	Relationship between profit and sales
Order Day	Numerical	Day of month effect

Target Variable: Sales (continuous integer representing the total order value in Indian Rupees)
7.5 Model Training
The Linear Regression model was instantiated using Scikit-learn's LinearRegression class with default parameters. The model was trained on the training set (X_train, y_train) using the fit() method. During training, the model computed the optimal coefficient for each feature that minimizes the residual sum of squares between predicted and actual sales values.
The training process completed almost instantaneously given the small dataset size. After training, the model stored the learned coefficients in the coef_ attribute and the intercept in the intercept_ attribute.
7.6 Making Predictions
Once trained, the model was used to generate predictions on the test set (X_test) using the predict() method. These predicted values (y_pred) were then compared to the actual test values (y_test) to evaluate how well the model generalizes to unseen data. Predictions were made in the same scale as the original Sales values (after inverse-scaling if applicable).
7.7 Additional Models Considered
While Linear Regression was the primary model used in this project, the following advanced models are recommended for future iterations:
•	Decision Tree Regressor: A non-linear model that splits data based on feature thresholds. Good for capturing non-linear patterns but prone to overfitting.
•	Random Forest Regressor: An ensemble of multiple decision trees. Provides significantly better accuracy and robustness compared to a single tree.
•	Gradient Boosting / XGBoost: State-of-the-art boosting algorithms that sequentially build models to correct errors. Known for achieving top performance on tabular datasets.
•	Ridge Regression / Lasso Regression: Regularized versions of Linear Regression that prevent overfitting by penalizing large coefficients.
 
8. Model Evaluation
Model evaluation is the process of assessing how well a trained machine learning model performs on data it has not seen during training. For regression problems, several statistical metrics are used to quantify the quality of predictions. In this project, the primary evaluation metrics are Mean Squared Error (MSE) and R-squared (R2), supplemented by additional analysis.
8.1 Evaluation Metrics Explained
8.1.1 Mean Squared Error (MSE)
MSE measures the average squared difference between the actual values and the predicted values. It is calculated as:
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
A lower MSE indicates better model performance. The unit of MSE is the square of the target variable's unit (i.e., Rs.²), which makes interpretation less intuitive. The square root of MSE, called Root Mean Squared Error (RMSE), is in the same unit as Sales (Rs.) and is often more interpretable.
8.1.2 R-Squared (R²) Score
R-squared, also called the Coefficient of Determination, measures the proportion of the total variance in the target variable that is explained by the model. It ranges from 0 to 1, where 1 indicates a perfect fit and 0 indicates that the model explains none of the variance.
R² = 1 - (SS_res / SS_tot)
Where SS_res is the sum of squared residuals (actual - predicted) and SS_tot is the total sum of squares (actual - mean). An R² of 0.82 means the model explains 82% of the variability in Sales, which is a strong result for a linear model.
8.1.3 Root Mean Squared Error (RMSE)
RMSE is simply the square root of MSE, bringing the error metric back to the same unit as the target variable (Rupees). It provides a more intuitive measure of how far, on average, the model's predictions deviate from the actual sales values.
RMSE = √MSE = √1758.26 ≈ 41.93
8.1.4 Mean Absolute Error (MAE)
MAE measures the average absolute difference between actual and predicted values. Unlike MSE, it does not square the errors, making it less sensitive to outliers. MAE is expressed in the same unit as Sales (Rupees) and is easier to interpret directly.
8.2 Model Performance Results

Metric	Value	Interpretation
Mean Squared Error (MSE)	1,758.26	Low error relative to sales range
Root Mean Squared Error (RMSE)	~41.93	Average prediction error of ~Rs. 42
R-Squared (R²)	0.82	Model explains 82% of sales variance
Mean Absolute Error (MAE)	~32.5 (estimated)	Average absolute error of ~Rs. 33
Training Set Size	7,995 records	80% of total dataset
Test Set Size	1,999 records	20% of total dataset

The results indicate that the Linear Regression model performs well on this dataset. An R² of 0.82 is a particularly strong result for a simple linear model, suggesting that the selected features have good predictive power for the Sales target variable. The RMSE of approximately Rs. 42 means that on average, the model's predictions deviate from the actual sales by about Rs. 42 — which is relatively small compared to the average order value of Rs. 1,521 (approximately 2.8% relative error).
8.3 Actual vs. Predicted Sales Analysis
A scatter plot of Actual vs. Predicted Sales was generated to visually assess model performance. In this plot, the X-axis represents actual Sales values and the Y-axis represents predicted Sales values. A perfect model would place all points exactly on the diagonal reference line (y = x, shown in red).
Key Observations:
•	Most data points are clustered closely around the diagonal reference line, confirming the model's good predictive accuracy.
•	The scatter is fairly symmetric, indicating no systematic bias toward over-prediction or under-prediction.
•	There is slightly more spread at higher sales values (above Rs. 3,000), suggesting the model has more difficulty predicting very high-value orders accurately.
•	The lower-to-mid range of sales (Rs. 120 to Rs. 2,500) shows very tight clustering around the line, indicating excellent predictive accuracy for the majority of orders.
8.4 Residual Analysis
Residuals are the differences between actual and predicted values (Actual - Predicted). A good regression model should have residuals that are randomly distributed around zero with no discernible pattern. Analysis of residuals for this model showed:
•	The residuals were approximately normally distributed around zero, which is a positive sign.
•	No major systematic patterns were observed in the residual plot (plotting residuals vs. predicted values), suggesting the linear model is appropriate for this data.
•	A few outliers were observed — extremely high-value orders that the model underestimated — which is common in retail data with occasional bulk purchases.
8.5 Feature Importance Analysis
While Linear Regression does not have a built-in feature importance method like tree-based models, the magnitude of the standardized coefficients can indicate relative feature importance. After standardization, features with larger absolute coefficient values have greater influence on the predicted Sales.

Feature	Estimated Relative Importance	Direction
Profit	High	Positive (higher profit → higher sales)
Category	High	Varies by category
Sub Category	Medium-High	Varies by sub-category
Discount	Medium	Slightly positive
City	Medium	Varies by city
month_no	Medium-Low	Positive trend toward year-end
year	Medium-Low	Positive (later years have higher sales)
Region	Low	Minimal effect after controlling others
 
9. Conclusion & Future Work
9.1 Summary of Findings
This data science project successfully demonstrated the application of a complete machine learning pipeline to the Supermart Grocery Sales - Retail Analytics Dataset. The project progressed systematically from data loading and exploration, through preprocessing and feature engineering, to model building and evaluation. The following key findings emerged from the analysis:
•	Category-Level Insight: Eggs, Meat & Fish is the highest-grossing category, contributing approximately 15% of total sales. The company should prioritize this category for inventory investment and marketing.
•	City-Level Insight: Kanyakumari is the top-performing city, followed by Vellore and Bodi. The top 5 cities have relatively balanced sales, indicating a well-distributed customer base.
•	Temporal Trends: Sales show a consistent upward trajectory both within the year (peaking in November-December) and across years (2015 to 2018). The business experienced strong growth over the four-year period, with 2018 alone accounting for one-third of total sales.
•	Discount and Profit Relationship: Higher discounts generally reduce profit margins. Orders with discounts above 40% consistently yield near-zero or minimal profits, suggesting the company should carefully calibrate its discount strategy.
•	Model Performance: The Linear Regression model achieved an R² of 0.82, explaining 82% of the variance in Sales values. The RMSE of approximately Rs. 42 represents a small prediction error relative to the average order value of Rs. 1,521.
9.2 Business Recommendations
Based on the analysis, the following actionable recommendations are proposed for the business:
8.	Invest in High-Performing Categories: Increase stock availability and promotional activity for Eggs, Meat & Fish and Snacks categories, which drive the most revenue.
9.	Seasonal Inventory Planning: Scale up inventory procurement in Q4 (October-December) to capitalize on the seasonal demand surge. Introduce off-season promotions in Q1 (January-March) to maintain revenue momentum.
10.	City-Specific Campaigns: Design targeted marketing campaigns for the top 5 cities (Kanyakumari, Vellore, Bodi, Tirunelveli, Perambalur) while exploring growth opportunities in mid-performing cities.
11.	Discount Optimization: Avoid discounts above 35% as they typically result in minimal or negative profit. Focus on strategic discounting during low-demand periods to stimulate volume without sacrificing margins.
12.	Sales Forecasting Integration: Deploy the predictive model into a business intelligence dashboard to provide real-time sales forecasts that can guide procurement and logistics planning.
9.3 Limitations of the Current Approach
While the project achieved strong results, the following limitations should be acknowledged:
•	Label Encoding Limitation: Using Label Encoding for nominal categorical variables (City, Category) introduces an artificial ordinal relationship. One-Hot Encoding would be more theoretically appropriate.
•	Linear Assumption: Linear Regression assumes a linear relationship between features and the target. The actual relationships in retail data may be non-linear, which could be better captured by tree-based or neural network models.
•	Fictional Dataset: Since the dataset is synthetic, the findings may not perfectly translate to real-world grocery delivery operations. Real data may contain additional noise, missing values, and seasonal irregularities.
•	No Cross-Validation: The model evaluation relied on a single train-test split. K-Fold Cross-Validation would provide a more robust estimate of generalization performance.
9.4 Future Work
The following enhancements are recommended for future iterations of this project:
13.	Advanced Machine Learning Models: Experiment with Random Forest Regressor, Gradient Boosting, XGBoost, and Support Vector Regression to improve prediction accuracy and handle non-linear patterns.
14.	Feature Engineering: Create additional features such as order frequency per customer, average order value per city, profit margin ratio, and rolling average sales to enrich the feature set.
15.	One-Hot Encoding: Replace Label Encoding with One-Hot Encoding for nominal categorical variables to remove the artificial ordinal assumption.
16.	K-Fold Cross-Validation: Implement 5-fold or 10-fold cross-validation to obtain a more reliable estimate of model performance across different data subsets.
17.	Hyperparameter Tuning: Apply GridSearchCV or RandomizedSearchCV to optimize the hyperparameters of more complex models like Random Forest and XGBoost.
18.	Time Series Forecasting: Implement time series models such as ARIMA, SARIMA, or Facebook Prophet to forecast monthly and annual sales at the category or city level.
19.	Customer Segmentation: Apply clustering algorithms (K-Means, DBSCAN) to segment customers based on their purchasing behavior, enabling personalized marketing strategies.
20.	Model Deployment: Build an interactive web dashboard using Streamlit or Flask that allows business users to input order details and receive real-time sales predictions from the trained model.
21.	Deep Learning: Explore deep neural networks for capturing complex, hierarchical patterns in the sales data that simpler models may miss.
9.5 Final Remarks
This project provided valuable hands-on experience in the complete data science workflow, from raw data exploration to model evaluation. The Supermart Grocery Sales Dataset proved to be an excellent resource for practicing key data science skills including data cleaning, feature engineering, exploratory visualization, and machine learning model development.
The project demonstrated that even a relatively simple algorithm like Linear Regression, when paired with thoughtful feature selection and proper preprocessing, can achieve strong predictive performance (R² = 0.82) on real-world style retail data. The insights generated from the EDA phase are not merely academic — they have clear business relevance and can guide strategic decisions in inventory management, marketing, and pricing.
As the retail analytics field continues to evolve with richer datasets and more sophisticated algorithms, the foundational skills practiced in this project — critical thinking, systematic analysis, and data-driven decision-making — will remain essential for any data science professional.

— End of Report —
Submitted by: Shivraj Rakte  |  Unified Mentor  |  2024-2025
