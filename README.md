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
* <b>Level:</b> the baseline value for the series
* <b>Trend:</b> Linear increasing or decreasing behavior of the serie over time
* <b>Seasonality:</b> Repeating patterns or cycles of behavior over time.
* <b>Noise:</b> Variability in the observations that cannot be explained by the model

The reunion of these 4 components is equal to the time serie, it can be additive if:

<p align="center"> <b>y = level + trend + seasonality + noise</b> </p>

or multiplicative if:

<p align="center"> <b>y = level * trend * seasonality * noise</b> </p>

In our case the time serie is multiplicative.

<p align="center"> <b>Bitcoin's Price since 2014</b>
<img src="https://user-images.githubusercontent.com/65224852/137589114-bf28c5be-3210-4ebe-b76a-25a5d44fc34f.png">
</p>

<p align="center"> <b>Bitcoin's Trend since 2014</b>
<img src="https://user-images.githubusercontent.com/65224852/137589120-86e8e6bb-d8b7-486c-91cb-9fe7d2cc132a.png">
</p>

We can clearly see that the trend is constantly rising, with a peak toward 2021.

The trend is a centered moving average, first we remove the mean by dividing each individual value by the series mean:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/137590297-3f2141d4-e763-4cf6-b3d2-43e1639dd7b3.PNG">
</p>

Then we calculate the moving average based on a sliding window *L* :

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/137590562-a3c91a8c-3654-499a-b1d5-18b62245142f.PNG">
</p>

(statsmodels doesn't exactly do the same calculations, though I suspect the way I describe it to be better)

<p align="center"> <b>Bitcoin's Seasonality since 2014</b>
<img src="https://user-images.githubusercontent.com/65224852/137592003-189d5162-1830-44ca-a075-c6ab8d410f52.png">
</p>

The seasonality indicates that there is a rising trend toward January and June and a decreasing trend in between.

Seasonality is the difference between the centered actual value and the trend:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/137873111-e6f5f9cc-d2b4-4247-93f8-63ca83aa2c78.PNG">
</p>

The network is made of two layers of bidirectionnal LSTM units with a 20 dense at the end in order to predict the next 20 values of the time serie.

## Project 2 - ARIMA

In this second project we will try to get useful statistics from the data by turning it into a stationary serie.

A stationary serie is a serie that does not depend on time (just like the stationary Schrödinger Equation or TISE where the wave function can be expressed only with the spatial part)

## Project 3 - Bitcoin forecast with multivariate time series

This third project aims to forecast the price of the Bitcoin using all the features in the dataset plus the values of others time series as well (such as Ethereum price).
