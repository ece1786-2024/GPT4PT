# GPT-4 Agents Prompts

Comparing with GPT-2 or GPT-3.5 versions, GPT-4 has an unique advantage that it's able to not only classify the MBTI attributes, but also gives the trend of how much a person leaning towards an attribute. The trend is a difference concept of the confidence on the result, for example, the model can be 90% sure that a person is 75% Introvert and 25% Extrovert, which means their decisions are mostly aligned with introvert type but also somewhat extrovert. It's like you are **pretty sure** (*model* *confidence*) that you want to drink cold cola (*result* *type*) but don't want **too much** (*result* *trend*) ice.

Therefore, this module introduces the prompts to reproduce GPT-4 agents that can specialized in classifying the user's personality from `0% confidence in 0% I/S/T/J/-A 0% E/N/F/P/-T` to `[*Threshold*]% confidence in [*Percentage*]% I/S/T/J/-A [*Percentage*]% E/N/F/P/-T`.


# Master agent

## Objective

To orchestrate 5 specialized agents, which interact with the user, and take user inputs and results from the specialized agents to generate the “next question” and final report.

# Specialized agents

## **Introvert (I) vs. Extrovert (E)**

### Objective

To determine the confidence and breakdown of I vs. E. For example, 90% confidence in 75% E / 25 % I.

## Sensing (S) vs. Intuition (N)

### Objective

To determine the confidence and breakdown of S vs. N. For example, 90% confidence in 75% S / 25 % N.

## Thinking (T) vs. Feeling (F)

### Objective

To determine the confidence and breakdown of T vs. F. For example, 90% confidence in 75% T / 25 % F.

## Judging (J) vs. Perceiving (P)

### Objective

To determine the confidence and breakdown of J vs. P. For example, 90% confidence in 75% J / 25 % P.

## Assertive (-A) vs. Turbulent (-T.)

### Objective

To determine the confidence and breakdown of A vs. T. For example, 90% confidence in 75% A / 25 % T.

# Combined agent

### Objective

To determine the complete MBTI type by interactive questions. For example: 90% confidence in ESTP-T type.
