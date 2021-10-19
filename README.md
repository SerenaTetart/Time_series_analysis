# Time_series_analysis

## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Project 1: Bitcoin Forecast with LSTM](#project-1---Bitcoin-Forecast-with-LSTM)
* [Project 2: ARIMA](#Project-2---ARIMA)
* [Project 3: Bitcoin forecast with multivariate time series](#Project-3---Bitcoin-forecast-with-multivariate-time-series)

## General info
In this repository we will be learning how to analyse a time serie and forecast it's values.

## Requirements

The basic libraries for machine learning and deep learning 😃

Libraries:
* Numpy
* Tensorflow
* Keras
* Pandas

## Project 1 - Bitcoin forecast with LSTM

The dataset used is the market price everyday since 2014, I shared it in the repository.

A time serie is composed of 4 components:
* <b>y:</b> the baseline value for the series
* <b>Trend:</b> Linear increasing or decreasing behavior of the serie over time
* <b>Seasonality:</b> Repeating patterns or cycles of behavior over time.
* <b>Noise:</b> Variability in the observations that cannot be explained by the model

The reunion of these 4 components is equal to the time serie, it can be additive if:

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
<img src="https://user-images.githubusercontent.com/65224852/137881530-f5e11af9-661a-46e6-9943-573de48c4e52.PNG">
</p>

<p align="center"> <b>Bitcoin's Seasonality since 2014</b>
<img src="https://user-images.githubusercontent.com/65224852/137592003-189d5162-1830-44ca-a075-c6ab8d410f52.png">
</p>

The seasonality indicates that there is a rising trend toward January and June and a decreasing trend in between.

In order to get the seasonality and the noise we need to refer to the multiplicative formula in the beginning:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/137880794-24879d5d-4e8b-45f7-bb43-b11e8fd6ea31.PNG">
</p>

Then in order to isolate the seasonality we calculate the moving average based on one year, L=365 (could be one month) of our last formula:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/137884804-207fd775-30ec-41a4-8a0e-2355c2eee62b.PNG">
</p>

The network is made of two layers of bidirectionnal LSTM units with a 20 dense at the end in order to predict the next 20 values of the time serie.

## Project 2 - ARIMA

In this second project we will try to get useful statistics from the data by turning it into a stationary serie.

A stationary serie is a serie that does not depend on time (just like the stationary Schrödinger Equation or TISE where the wave function can be expressed only with the spatial part)

## Project 3 - Bitcoin forecast with multivariate time series

This third project aims to forecast the price of the Bitcoin using all the features in the dataset plus the values of others time series as well (such as Ethereum price).
