# Time_series_analysis

## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Introduction](#introduction---Quick-overview-of-time-series)
* [Project 1: Sales forecasting with ARIMA](#project-1---Sales-forecasting-with-ARIMA)
* [Project 2: Bitcoin forecasting with LSTM](#project-2---Bitcoin-forecasting-with-LSTM)
* [Project 3: BTC and ETH forecasting with multivariate time series](#Project-3---BTC-and-ETH-forecasting-with-multivariate-time-series)

## General info
In this repository you'll learn how to analyse a time serie and forecast it's values.

Here we will try to forecast sales and cryptocurrencies using ARIMA, LSTM and multivariate time series.

Because cryptocurrencies are highly volatile and doesn't follow a specific pattern the results won't be good for the last two projects.

## Requirements

The basic libraries for machine learning and deep learning 😃

Libraries:
* Numpy
* Tensorflow
* Keras
* Pandas
* statsmodels
* pmdarima

## Introduction - Quick overview of time series

For the purpose of this introduction we'll use as dataset Bitcoin's closing market price everyday since 2014.

A time serie is composed of 3 components:
* (<b>y:</b> the baseline value for the series)
* <b>Trend:</b> Linear increasing or decreasing behavior of the serie over time
* <b>Seasonality:</b> Repeating patterns or cycles of behavior over time.
* <b>Noise:</b> Variability in the observations that cannot be explained by the model

The reunion of these 3 components is equal to the time serie, it can be additive if:

<p align="center"> <b>y = trend + seasonality + noise</b> </p>

or multiplicative if:

<p align="center"> <b>y = trend * seasonality * noise</b> </p>

In our case the time serie is multiplicative.

<p align="center"> <b>Bitcoin's Price since 2014</b>
<img src="https://user-images.githubusercontent.com/65224852/137589114-bf28c5be-3210-4ebe-b76a-25a5d44fc34f.png">
</p>

<p align="center"> <b>Bitcoin's Trend since 2014</b>
<img src="https://user-images.githubusercontent.com/65224852/137589120-86e8e6bb-d8b7-486c-91cb-9fe7d2cc132a.png">
</p>

We can clearly see that the trend is constantly rising, with a peak toward 2021.

The Trend is calculated based on the moving average with a sliding window *L* :

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/145617459-3c80c147-e2e0-4690-9053-30eb04087c0e.png">
</p>

<p align="center"> <b>Bitcoin's Seasonality since 2014</b>
<img src="https://user-images.githubusercontent.com/65224852/137592003-189d5162-1830-44ca-a075-c6ab8d410f52.png">
</p>

The seasonality indicates that there is a rising trend toward January and June and a decreasing trend in between.

In order to get the seasonality and the noise we need to refer to the multiplicative formula in the beginning:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/145714059-7e9d4876-8fc0-473c-9614-edd347a91158.png">
</p>

Then in order to isolate the seasonality we calculate the moving average based on one year, **L=365** (could be one month) of our last formula:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/137884804-207fd775-30ec-41a4-8a0e-2355c2eee62b.PNG">
</p>

Finally we calculate the noise by dividing the seasonal noise by the seasonal component:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/137888988-d0bca21d-7793-4600-aacc-31dcb7f9b144.PNG">
</p>

<p align="center"> <b>Bitcoin's Noise since 2014</b>
<img src="https://user-images.githubusercontent.com/65224852/151705528-f998011a-f8ec-4e25-942f-b0a868def041.png">
</p>

As we can see the data is very noisy, this time serie depends on a lot of factors.

## Project 1 - Sales forecasting with ARIMA

In this first project we will forecast the sales and number of orders of a retail store with the model ARIMA which stands for ‘Auto Regressive Integrated Moving Average’.

The dataset is from Kaggle: <a href='https://www.kaggle.com/rohitsahoo/sales-forecasting'>Superstore Sales Dataset</a>

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/151908222-69764ca2-3d42-4663-897c-aa8b249f1e02.png">
</p>

## Project 2 - Bitcoin forecasting with LSTM

For this project we'll use as dataset Bitcoin's closing market price everyday since 2014.

The network is made of two layers of bidirectionnal LSTM units with a 20 dense at the end in order to predict the next 20 values of the time serie.

LSTMs are great at learning from long-term dependencies on sequences of data, when made bidirectionnal they also train on a reversed copy of the input sequence, this can provide additional context to the network and result in faster and even fuller learning on the problem.

Sadly the forecasts are quite imprecise, this is because there is a lot of noises and actions or cryptocurrencies price depends on a lot of factors that a time serie alone can not represent.

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/145723484-b8a62815-4b56-4c92-938e-d56f63c681de.png">
</p>

Though the model did manage to learn a rising trend and is not totally wrong.

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/137901708-13a1cc36-80e4-448d-839d-5cd6ef674939.png">
</p>

This last forecast is wrong.

## Project 3 - BTC and ETH forecasting with multivariate time series

This third project aims to forecast the price of cryptocurrencies using all the features in the dataset, and combining two cryptocurrencies to see if the results are better.

The LSTM overfits so quickly that we need to set the number of epochs to 12 (batch size also influence a lot the number of epochs required), furthermore if the number of features is too big we need to set the loss to a MSLE (mean squared logarithmic error) so that big differences count as much as little ones.

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/145723078-b9311f65-b83d-48d2-aa31-19d03077dd37.png">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/145723079-0a45d1d5-d48a-4e44-82c3-c4482b8769f0.png">
</p>

But nope this is still too bad.
