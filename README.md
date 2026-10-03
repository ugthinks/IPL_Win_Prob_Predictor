# 🏏 IPL Win Probability Predictor

A machine learning based web application that predicts the **win probability of both teams during an IPL chase** based on the current match situation.

The application uses **Logistic Regression** and is built with **Python, Scikit-learn and Streamlit**.

---

## Overview

The idea is simple:

Given the current situation of an IPL match, such as:

- Target score
- Current score
- Wickets lost
- Overs completed
- Chasing team
- Bowling team
- Venue

the model estimates the probability of:

- 🏏 Chasing team winning
- 🎯 Bowling team winning

The prediction is based on historical IPL match data.

---

## Example

Suppose the match situation is:

```text
Target:          190
Current Score:   100
Wickets Lost:    2
Overs:           12.0