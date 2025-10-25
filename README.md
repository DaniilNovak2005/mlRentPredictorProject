# Predicting Profit from Renting houses using XG-Boost

## ⚠️ WARNING!

This project involves the use of webscraping, I am not responsible for any misconduct using the webscrapers I have provided, and my page does not give away any internal data. It only provides theoretical fictional examples to run locally. If you have any legal inquiries, please contact me at the email provided at the bottom. This project is also heavily WIP, and has edge cases where the prediction model breaks due to the varied nature of the house market. The tutorial is run on a real life example that has been randomly chosen.

## ⚠️ Status: Work in Progress

This project is in active development. The current model and analysis are **limited to data from Florida only**. Future work will focus on expanding data collection to other states and improving model accuracy.

## 1. Problem & Motivation

Rental Prices in Florida have been notouriously difficult to predict due to a variety of different factors, including calculating dynamic insurance costs, understanding the relationship between amenities and nearby houses, and more. Even though there are already existing tools from Zillow to calculate expected profit, it is innaccurate and doesn't provide enough statistics about the nature of the data being used and insurance info, because of the nature of the data's privacy. I'm trying to provide an application which allows the user to use real-time analytics from scraped data to identify the most profitable areas and view important metrics, such as Cash-on-Cash return, which is essentially the expected profit in terms of percentage difference from the costs.

## 2. Methodology

My process consists of three main parts:

### A. Data Collection (Web Scraper)

To do this I first created a docker image with the neccessary dependencies, Python3, Nodriver, and others listed in the project. I then ran my script, which used a virtual chrome driver to quickly scrape and navigate Zillow pages for every house link, with the help of rotating proxy services. With these URLs, I then fetched the details for each house in the scraped list, noting important events and statistics. I finally ran feature-engineering scripts to help improve the correlation and accuracy of the data.

### B. Model Training

After collection I trained a XG-Boost Model on my dataset. This involved first pre-processing the data. I one-hot encoded variables such as parking and housing features, and created further analytics like average zipcode Price Estimate and average zipcode median income. Using this data I then did a 80-10-10 cross evaluation split, training the XG-Boost model with an objective to minimize R2 score. For the training, I made 2 models, a Days on Market predictor and a Rent Price predictor.

My personal results were as follows:

#### Performance Days Prediction Model:

Mean cross-validation R2 score: 0.3132883618405109

Mean Absolute Error (MAE): 34.41512029528819

Mean Squared Error (MSE): 2365.677451849198

Root Mean Squared Error (RMSE): 48.63823035277083

![Alt text](https://i.imgur.com/sKYcL3j.png "Prediction Scatterplot")
![Alt text](https://i.imgur.com/P7Jk7ft.png "Prediction Variable Correlation Plot")

#### Performance Price Prediction Model:

Mean cross-validation R2 score: 0.8113493059818317

Mean Absolute Error (MAE): 298.1691469733721

Mean Squared Error (MSE): 277305.0135186118

Root Mean Squared Error (RMSE): 526.5975821427704

![Alt text](https://i.imgur.com/bp0jCpf.png "Prediction Scatterplot")
![Alt text](https://i.imgur.com/s5GVbK3.png "Prediction Variable Correlation Plot")

My results yielded results which showed a very great amount of variance in the predicted number of days, with an amount of predictability. My R2 score of .313 was good in the context of the market, but of course not accurate enough. In order to provide a metric which could be accurate and useful, I then trained a binary classification XGBoost model, to predict whether a house would be sold within 30 days or not, with the following results.

#### Fast Sale classification model:

Accuracy: 0.7519

Precision: 0.6720

Recall: 0.6901

F1-score: 0.6809

ROC AUC: 0.8389

Confusion Matrix:

array([[2311,  613],
       [ 564, 1256]])

Based off the price predictions, I was able to create a strong starting gradient for the project, with exceptional accuracy on the lower cost models, with a large amount of uncertainty working with expensive housing. This is due to the fact that at the prices these homes are at, the renting profit is expected to be very low due to the houses price.

### C. Web App

I built a simple 3 level full stack application to actually create interactive reports with the data. In the attached flask apps it is possible to predict real time housing analytics through Zelle, as well as manually input the data. The returned report is a compilation of the following data:

- Expected Rental Price
- Toggleable Insurance Situations
- Expected ROI
- How long it will stay on the market, and how it compares to other nearby properties.
- Rent Price trends in nearby zipcodes.
- Interactive Menu to customize house data, adding speculative what if scenarios.

The user is able to use the React frontend to create basic report requests using their accounts, and communicate with the Express.js middleman server. This middleman server is responsible for saving the user and report data in a MongoDB database. It also handles each users requests, communicating with the backend machine server. The backend machine server handles the models loading and fitting. It takes the users requests and scrapes the Zillow data in real time, or train off existing data. It then outputs a myriad of statistics and info, that is loaded on the users frontend.

## 3. Limitations & Next Steps

Currently, the React frontend and model is unfinished. This is due to a lack of time on behalf of my part due to a busy college schedule. The data is enclosed to Florida only, and insurance data is innaccurate due to the lack of internal insurance company data available. The project is still under heavy development, but the finished product would be a fully functioning web application, and multiple models for each region of the USA.

## 4. Run Locally

#### Web Scraper

1. Tests are provided in the tests folder for the scraper. Each test launches a part of the scraping process, proxies are required for intensive scraping, which are provided by online sources.
2. Install the requirements.txt.
3. Run the target test using python3.

#### App

##### Front end

1. Enter the frontend's src folder.
2. Run npm i to install the required packages
3. Run npm run dev to start the server.

##### Middleman

1. Enter the middlemans's src folder, backend.
2. Run npm i to install the required packages
3. Replace .ENV variables with new ones, including the target SFTP server, JWT token, and MongoDB database.
4. Run npm run start to start the server.

##### Back end Machine

1. Enter the backends's src folder.
2. Replace the data with whatever new data you may need (NOTE: Insurance is calculated using open source data and may be innaccurate for regions outside FL.)
3. Install requirements.txt
4. Run the app.py script.

A tutorial is provided below:
https://youtu.be/dN7VxeoUCTo?si=HWTNM5M0q3-_gyX-

## 5. Contact

For any questions or inquiries, contact me :)
daniilnovak@ucsb.edu
